from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
from collections import deque

from einops import rearrange, repeat
import dataclasses
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

# for visualization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head, Slicer, SpecialHead, DiscreteDist
from .tokenizer import Tokenizer

from .transformer import Transformer, TransformerConfig, get_sinusoid_encoding_table
from .transformer import Perceiver, PerceiverConfig
from .reward_utils import losses_dict

from .world_model_env import MAWorldModelEnv
from utils import init_weights, action_split_into_bins, discretize_into_bins, initialize_weights, symlog, symexp
import wandb
import ipdb

@dataclass
class MAWorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    pred_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    pred_avail_action: torch.FloatTensor
    attn_output: List


class MAWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, num_action_tokens: int, num_agents: int,
                 config: TransformerConfig, perattn_config: PerceiverConfig,
                 action_dim: int, use_bin: bool = False, bins: int = 64,
                 ### options for setting prediction head
                 use_symlog: bool = False, use_ce_for_end: bool = False, use_ce_for_av_action: bool = True, 
                 use_ce_for_reward: bool = False, rewards_prediction_config: dict = None) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.use_bin = use_bin
        self.bins = bins
        
        self.num_modalities = 3

        self.config = config
        self.num_agents = num_agents

        ## perceiver attention
        self.perattn_config = perattn_config
        self.perattn = Perceiver(**dataclasses.asdict(perattn_config))
        self.agent_id_pos_emb = get_sinusoid_encoding_table(30, perattn_config.dim)
        # self.agent_id_pos_emb = nn.Embedding(num_agents, perattn_config.context_dim)
        ## --------------------

        self.num_action_tokens = num_action_tokens  # for continuous task, this should be dimension of joint action (e.g. like ManiSkill2)
        self.num_obs_tokens = config.tokens_per_block - num_action_tokens - 1  # 其中有一个是perceiver attn的输出

        self.transformer = Transformer(config)

        act_tokens_pattern = torch.zeros(config.tokens_per_block)
        # act_tokens_pattern[-num_action_tokens:] = 1 
        act_tokens_pattern[-1 - num_action_tokens : -1] = 1   ### modified at 0326
        self.act_tokens_pattern = act_tokens_pattern

        obs_tokens_pattern = torch.zeros(config.tokens_per_block)
        obs_tokens_pattern[:self.num_obs_tokens] = 1
        self.obs_tokens_pattern = obs_tokens_pattern
        
        ### for autoregressive manner
        obs_autoregress_pattern = obs_tokens_pattern.clone()
        obs_autoregress_pattern = torch.roll(obs_autoregress_pattern, -1)

        ### due to attention mask, the last token of transformer output is generated by all tokens of input
        all_but_last_pattern = torch.zeros(config.tokens_per_block)
        all_but_last_pattern[-1] = 1

        ### Perceiver Attention output pattern
        perattn_pattern = torch.zeros(config.tokens_per_block)
        # perattn_pattern[-num_action_tokens - 1 : -num_action_tokens] = 1
        perattn_pattern[-1] = 1
        self.perattn_pattern = perattn_pattern

        # self.obs_embeddings_slicer = Slicer(max_blocks=config.max_blocks, block_mask=obs_autoregress_pattern)
        self.perattn_slicer = Slicer(max_blocks=config.max_blocks, block_mask=perattn_pattern)

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )


        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=obs_autoregress_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        ## 加入适合dense的reward predictor
        self.use_symlog = use_symlog  # whether to use symlog transformation
        self.use_ce_for_reward = use_ce_for_reward
        if use_ce_for_reward:
            print("Use cross-entropy to train the prediction of reward...")
        else:
            print("Use SmoothL1Loss to train the prediction of reward...")

        if not self.use_ce_for_reward:
            self.head_rewards = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, 1),
                )
            )

        else:
            assert rewards_prediction_config is not None
            self.use_symlog = True
            bin_width = (rewards_prediction_config["max_v"] - rewards_prediction_config["min_v"]) / rewards_prediction_config["bins"]
            self.reward_loss = losses_dict[rewards_prediction_config["loss_type"]](
                min_value = rewards_prediction_config["min_v"],
                max_value = rewards_prediction_config["max_v"],
                num_bins = rewards_prediction_config["bins"],
                sigma = bin_width * 0.75
            )
            print(f'Use {self.reward_loss} for discrete labels...')

            self.head_rewards = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, self.reward_loss.output_dim),
                )
            )
        

        self.use_ce_for_end = use_ce_for_end
        if use_ce_for_end:
            print("Use cross-entropy to train the prediction of termination...")
        else:
            print("Use log-prob to train the prediction of termination...")

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2 if use_ce_for_end else 1),
            )
        )

        self.action_dim = action_dim
        self.use_ce_for_av_action = use_ce_for_av_action
        ## 注意这个avail_actions预测的是下一时刻的avail_actions
        if use_ce_for_av_action:
            print("Use cross-entropy to train the prediction of av_action...")
        else:
            print("Use log-prob to train the prediction of av_action...")

        if not self.use_ce_for_av_action:
            self.heads_avail_actions = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, action_dim),
                )
            )
        
        else:
            self.heads_avail_actions = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=DiscreteDist(
                    config.embed_dim, self.act_vocab_size, 2, 256
                )
            )

        # info_loss head
        # self.head_last_action = Head(
        #     max_blocks=config.max_blocks,
        #     block_mask=all_but_last_pattern,
        #     head_module=nn.Sequential(
        #         nn.Linear(config.embed_dim, config.embed_dim),
        #         nn.ELU(),
        #         nn.Linear(config.embed_dim, config.embed_dim),
        #         nn.ELU(),
        #         nn.Linear(config.embed_dim, action_dim),
        #     )
        # )

        self.apply(init_weights)
        
        # initialize_weights(self.head_rewards, mode='xavier')
        # initialize_weights(self.head_ends, mode='xavier')
        # initialize_weights(self.heads_avail_actions, mode='xavier')
        
        self.use_ib = False # use iris databuffer
        if self.use_symlog:
            print("Enable `symlog` to transform the reward targets...")
        else:
            print("Disable `symlog` to transform...")


    def __repr__(self) -> str:
        return "multi_agent_world_model"

    def forward(self, tokens: torch.LongTensor, perattn_out: torch.Tensor = None, past_keys_values: Optional[KeysValues] = None, return_attn: bool = False, attention_mask: torch.Tensor = None) -> MAWorldModelOutput:
        bs = tokens.size(0)
        num_steps = tokens.size(1)  # (B, T)

        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        
        sequences = self.embedder(tokens, num_steps, prev_steps)

        indices = self.perattn_slicer.compute_slice(num_steps, prev_steps)
        if perattn_out is not None:
            assert len(indices) != 0
            sequences[:, indices] = perattn_out
        else:
            assert len(indices) == 0

        sequences += self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x, attn_output = self.transformer(sequences,
                                          past_keys_values,
                                          return_attn = return_attn,
                                          attention_mask = attention_mask)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)

        # logits_last_action = self.head_last_action(x, num_steps=num_steps, prev_steps=prev_steps)
        # logits_last_action = rearrange(logits_last_action, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        pred_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        ## with new special head
        # pred_rewards = self.head_rewards(x, perattn_out, num_steps=num_steps, prev_steps=prev_steps)

        # pred_rewards = rearrange(pred_rewards, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        ## with new special head
        # logits_ends = self.head_ends(x, perattn_out, num_steps=num_steps, prev_steps=prev_steps)

        # logits_ends = rearrange(logits_ends, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        logits_avail_action = self.heads_avail_actions(x, num_steps=num_steps, prev_steps=prev_steps)
        ## with new special head
        # logits_avail_action = self.heads_avail_actions(x, perattn_out, num_steps=num_steps, prev_steps=prev_steps)

        # logits_avail_action = rearrange(logits_avail_action, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        return MAWorldModelOutput(x, logits_observations, pred_rewards, logits_ends, logits_avail_action, attn_output=attn_output)

    def compute_loss(self,
                     batch,
                     tokenizer: Tokenizer,
                     attention_mask: torch.Tensor = None,
                     **kwargs: Any):
        device = batch['observation'].device

        # only take discrete action space into account
        act_tokens = torch.argmax(batch['action'], dim=-1, keepdim=True)

        ### modified for ablation ###
        if not self.use_bin:
            with torch.no_grad():
                ### when tokenizer is `Tokenizer` run these two lines
                # tokenizer_encodings = tokenizer.encode(batch['observation'], should_preprocess=True)  # (B, L, K)
                # obs_tokens = tokenizer_encodings.tokens

                ### when tokenizer is `SimpleVQAutoEncoder` run these two lines
                obs_t_embeds, obs_tokens = tokenizer.encode(batch['observation'], should_preprocess=True)
                obs_tokens = obs_tokens.to(torch.long)
        else:
            obs_tokens = discretize_into_bins(batch['observation'], self.bins)
        ### --------------------- ###

        ### 将obs encodings和action encodings一起过perceiver attention
        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](act_tokens)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        b, l, N, M, e = input_encodings.shape

        agent_id_emb = repeat(self.agent_id_pos_emb[:, :self.num_agents], '1 n e -> (b l) (n m) e', b = b, l = l, m = M)
        input_encodings = rearrange(input_encodings, 'b l n m e -> (b l) (n m) e') + agent_id_emb.detach().to(device)

        perattn_out = self.perattn(input_encodings)
        perattn_out = rearrange(perattn_out, '(b l) n e -> (b n) l e', b=b, l=l, n=N)

        # tokens = torch.cat([obs_tokens, torch.empty_like(act_tokens, device=device, dtype=torch.long), act_tokens], dim=-1) # (B, L, (K+N))
        tokens = torch.cat([obs_tokens, act_tokens, torch.empty_like(act_tokens, device=device, dtype=torch.long)], dim=-1) # (B, L, (K+N))
        tokens = rearrange(tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')  # (B, L(K+N))

        outputs = self(tokens, perattn_out = perattn_out, attention_mask = attention_mask)

        # compute labels
        if self.use_ib:  # if use iris databuffer
            valid_mask = batch['filled'].clone().unsqueeze(-1).expand(-1, -1, self.num_agents).to(torch.float32)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['reward'], batch['done'], batch['filled'])
            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b l o -> (b l) o')

            loss_obs = F.cross_entropy(logits_observations, labels_observations)

            if not self.use_classification:
                pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs.logits_ends), 1)
                loss_ends = -(pred_ends.log_prob((1. - labels_ends)) * valid_mask).sum() / valid_mask.sum()
            else:
                raise NotImplementedError

            l1_criterion = nn.SmoothL1Loss(reduction="none")

            ## regression label for rewards
            labels_rewards = symlog(batch['reward'])

            loss_rewards = l1_criterion(outputs.pred_rewards, labels_rewards)
            loss_rewards = (loss_rewards.squeeze(-1) * valid_mask).sum() / valid_mask.sum()

            # criterion = nn.CrossEntropyLoss(reduction='none')
            # logits_last_action = rearrange(outputs.logits_last_action[:, 1:], 'b l n a -> (b l n) a')
            # label_last_action = rearrange(batch['action'][:, :-1], 'b l n a -> (b l n) a')
            # fake = batch['filled'].unsqueeze(-1).expand(-1, -1, self.num_agents)
            # fake = rearrange(fake[:, :-1], 'b l n -> (b l n)')
            # info_loss = (criterion(logits_last_action, label_last_action.argmax(-1).view(-1)) * fake).mean()

            info_loss = 0.

            pred_av_actions = td.independent.Independent(td.Bernoulli(logits=outputs.pred_avail_action[:, :-1]), 1)
            loss_av_actions = -(pred_av_actions.log_prob(batch['av_action'][:, 1:]) * valid_mask[:, 1:]).sum() / valid_mask[:, 1:].sum()

        else:  # use mamba databuffer
            ### guided by dones mask, compute observation loss
            dones = rearrange(batch['done'], 'b l n 1 -> (b n) l')
            labels_obs_token = rearrange(obs_tokens, 'b l n m -> (b n) (l m)')
            loss_obs = 0.
            for idx in range(dones.size(0)):
                cur_done = dones[idx]
                if cur_done[:-1].sum() > 0:
                    done_idx = (cur_done == 1).nonzero().squeeze() + 1
                    divide_idx = done_idx * self.num_obs_tokens
                    cur_loss = F.cross_entropy(outputs.logits_observations[idx, :(divide_idx - 1)], labels_obs_token[idx, 1 : divide_idx]) + F.cross_entropy(outputs.logits_observations[idx, divide_idx:-1], labels_obs_token[idx, (divide_idx + 1):])
                    loss_obs += cur_loss / 2
                else:
                    loss_obs += F.cross_entropy(outputs.logits_observations[idx, :-1], labels_obs_token[idx, 1:])

            loss_obs /= dones.size(0)

            ### compute discount loss
            if not self.use_ce_for_end:
                pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs.logits_ends), 1)
                loss_ends = -torch.mean(pred_ends.log_prob((1. - rearrange(batch['done'], 'b l n 1 -> (b n) l 1'))))
            else:
                logits_ends = rearrange(outputs.logits_ends, 'b l e -> (b l) e')
                labels_ends = rearrange(batch['done'], 'b l n 1 -> (b n l)').to(torch.long)
                loss_ends = F.cross_entropy(logits_ends, labels_ends)

            ### compute reward loss
            labels_rewards = rearrange(batch['reward'], 'b l n 1 -> (b n) l 1')
            if self.use_symlog:
                labels_rewards = symlog(labels_rewards)
            
            if self.use_ce_for_reward:
                labels_rewards = rearrange(labels_rewards, 'b l 1 -> (b l 1)')
                logits_rewards = rearrange(outputs.pred_rewards, 'b l e -> (b l) e')
                loss_rewards = self.reward_loss(logits_rewards, labels_rewards)

            else:
                loss_rewards = F.smooth_l1_loss(outputs.pred_rewards, labels_rewards)

            ### compute av_action loss
            tmp = torch.roll(batch['done'], 1, dims=1).squeeze(-1)
            labels_av_actions = batch['av_action']
            labels_av_actions[tmp == True] = torch.ones_like(labels_av_actions[tmp == True], device=device)
            
            ## for cross-entropy loss
            if self.use_ce_for_av_action:
                logits_av_actions = rearrange(outputs.pred_avail_action[:, :-1], 'b l a e -> (b l a) e')
                labels_av_actions = rearrange(labels_av_actions, 'b l n e -> (b n) l e')[:, 1:].reshape(-1,).to(torch.long)
                loss_av_actions = F.cross_entropy(logits_av_actions, labels_av_actions)
            
            else:
                pred_av_actions = td.independent.Independent(td.Bernoulli(logits=outputs.pred_avail_action[:, :-1]), 1)
                labels_av_actions = rearrange(labels_av_actions, 'b l n e -> (b n) l e')
                loss_av_actions = -torch.mean(pred_av_actions.log_prob(labels_av_actions[:, 1:]))

            info_loss = 0.

        loss = loss_obs + loss_ends + loss_rewards + loss_av_actions + info_loss

        loss_dict = {
            'world_model/loss_obs': loss_obs.item(),
            'world_model/loss_rewards': loss_rewards.item(),
            'world_model/loss_ends': loss_ends.item(),
            'world_model/loss_av_actions': loss_av_actions.item(),
            'world_model/info_loss': 0.,
            'world_model/total_loss': loss.item(),
        }

        return loss, loss_dict
        # return LossWithIntermediateLosses(**loss_dict)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, filled: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(filled)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(obs_tokens), -100).transpose(1, 2), 'b n l k -> (b n) (l k)')[:, 1:]
        
        labels_rewards = rewards.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(rewards), 0.)

        labels_ends = ends.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(ends), 1.).to(torch.long)
        
        return labels_observations.reshape(-1), labels_rewards, labels_ends
    
    def compute_labels_world_model_all_valid(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        labels_observations = rearrange(obs_tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')[:, 1:]
        labels_rewards = rearrange(rewards.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        labels_ends = rearrange(ends.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1).to(torch.long)
    
    
    def get_perceiver_attn_out(self, obs_tokens, actions):
        device = obs_tokens.device
        shape = obs_tokens.shape
        
        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](actions)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        n, m, e = input_encodings.shape[-3:]
        input_encodings = rearrange(input_encodings, '... n m e -> (...) (n m) e')
        agent_id_emb = repeat(self.agent_id_pos_emb[:, :n], '1 n e -> b (n m) e', b = input_encodings.size(0), n = n, m = m)

        input_encodings += agent_id_emb.detach().to(device)
        perattn_out = self.perattn(input_encodings)

        perattn_out = perattn_out.reshape(*shape[:-1], -1)
        return perattn_out
    
    
    ### visualize attention map
    @torch.no_grad()
    def visualize_attn(self, sample, tokenizer, save_dir):
        # preliminary
        device = sample["observation"].device
        n_agents = sample['observation'].shape[-2]
        horizon = sample['observation'].shape[-3]
        obs_token_indices = rearrange(repeat(self.obs_tokens_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        obs_token_indices = (obs_token_indices == 1).nonzero().squeeze().numpy()
        act_token_indices = rearrange(repeat(self.act_tokens_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        act_token_indices = (act_token_indices == 1).nonzero().squeeze().numpy()
        perattn_indices = rearrange(repeat(self.perattn_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        perattn_indices = (perattn_indices == 1).nonzero().squeeze().numpy()
        
        save_dir.mkdir(parents=True, exist_ok=True)
        for agent_id in range(n_agents):
            tmp_dir = save_dir / f"agent_{agent_id}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
        
        _, obs_tokens = tokenizer.encode(sample['observation'], should_preprocess=True)
        obs_tokens = obs_tokens.to(torch.long)
        act_tokens = torch.argmax(sample['action'], dim=-1, keepdim=True)
        
        perattn_out = self.get_perceiver_attn_out(obs_tokens, act_tokens)
        b, l, n, e = perattn_out.shape
        perattn_out = rearrange(perattn_out, 'b l n e -> (b n) l e', b=b, l=l, n=n)
        
        tokens = torch.cat([obs_tokens, act_tokens, torch.empty_like(act_tokens, device=device, dtype=torch.long)], dim=-1)
        tokens = rearrange(tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')  # (B, L(K+N))

        outputs = self(tokens, perattn_out = perattn_out, return_attn=True)
        
        attn_output = outputs.attn_output
        
        # define custom cmap
        modality_colors = ["Oranges", "Greens", "Blues"]
        colors = []
        for color in modality_colors:
            cmap = mpl.colormaps[color]
            colors.append(
                cmap(np.linspace(0., 1., 333))
            )
            
        white_cmap = LinearSegmentedColormap.from_list("white", [(0., 'white'), (1., 'white')], N=1)
        colors.append(
            white_cmap(np.linspace(0., 1., 1))
        )
            
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", np.vstack(colors))
        red_cmap = mpl.colormaps["Oranges"]
        
        def save_matrix_as_image(matrix, filename, custom_cmap):
            plt.imshow(matrix, cmap=custom_cmap, vmin=0, vmax=1)
            # plt.colorbar(orientation="horizontal")
            
            plt.axis("off")
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=600)
            plt.close()
        
        ## save as image
        scale = 0.332
        for layer_id in range(len(attn_output)):
            attn_weight = attn_output[layer_id].cpu().numpy()
            attn_weight[:, :, obs_token_indices] *= scale
            
            attn_weight[:, :, perattn_indices] *= scale
            attn_weight[:, :, perattn_indices] += 0.333
            
            attn_weight[:, :, act_token_indices] *= scale
            attn_weight[:, :, act_token_indices] += 0.666
            
            attn_weight = np.where(np.tril(np.ones_like(attn_weight)) == 1, attn_weight, np.zeros_like(attn_weight) + 0.9995)
            
            for agent_id in range(attn_weight.shape[0]):
                for head_id in range(attn_weight.shape[1]):
                    save_matrix_as_image(attn_weight[agent_id, head_id],
                                         save_dir / f"agent_{agent_id}" / f"layer{layer_id}_head{head_id}.png",
                                         custom_cmap)
        
        print(f"Attention visualization has been saved to {str(save_dir)}.") 


def rollout_policy_trans(wm_env: MAWorldModelEnv, policy, horizons, observations, av_actions, filled, **kwargs):
    use_stack = kwargs.get("use_stack", False)

    init_obs = observations[:, -1].clone()
    av_action = av_actions[:, -1].clone()

    if use_stack:
        stack_obs_num = kwargs.get("stack_obs_num", None)
        assert stack_obs_num is not None and type(stack_obs_num) == int

        stack_obs = deque(maxlen=stack_obs_num)
        
        tmp_obs = observations[:, :-1].clone()
        tmp_filled = filled[:, :-1, None, None].clone().repeat(1, 1, *tmp_obs.shape[-2:])
        
        unvalid_obs = torch.zeros_like(tmp_obs, device=tmp_obs.device)
        tmp_obs = wm_env.tokenizer.encode_decode(tmp_obs, True, True)
        tmp_obs = torch.where(tmp_filled == True, tmp_obs, unvalid_obs)

        for index in range(stack_obs_num - 1):
            stack_obs.append(tmp_obs[:, index])
        
    actor_feats = []
    critic_feats = []
    actions = []
    av_actions = []
    policies = []
    rewards = []
    dones = []

    # initialize wm_env
    rec_obs, critic_feat = wm_env.reset_from_initial_observations(init_obs)

    for t in range(horizons):
        # critic_feat = rearrange(wm_env.world_model.embedder.embedding_tables[1](wm_env.obs_tokens), 'b n k e -> b n (k e)')

        # action, pi = policy(feat)
        if use_stack:
            stack_obs.append(rec_obs)
            feat = rearrange(torch.stack(list(stack_obs), dim=0), 'm b n e -> b n (m e)')

        else:
            feat = rec_obs

        action, pi = policy(feat)

        if av_action is not None:
            pi[av_action == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(av_action.squeeze(0))
        
        # actor_feats.append(feat)
        actor_feats.append(feat)
        policies.append(pi)
        actions.append(action)
        critic_feats.append(feat)

        rec_obs, reward, done, av_action, critic_feat = wm_env.step(torch.argmax(action, dim=-1).unsqueeze(-1), should_predict_next_obs=(t < horizons - 1))
        
        rewards.append(reward)
        dones.append(done)

    return {"actor_feats": torch.stack(actor_feats, dim=0), # torch.stack(actor_feats, dim=0),
            "critic_feats": torch.stack(critic_feats, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "discounts": torch.stack(dones, dim=0),
            }
        