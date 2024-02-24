from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange, repeat
import dataclasses
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

# from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head, Slicer
from .tokenizer import Tokenizer

from .transformer import Transformer, TransformerConfig, get_sinusoid_encoding_table
from .transformer import Perceiver, PerceiverConfig

from .world_model_env import MAWorldModelEnv
from utils import init_weights, action_split_into_bins, discretize_into_bins
import wandb
import ipdb

@dataclass
class MAWorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    pred_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    pred_avail_action: torch.FloatTensor


class MAWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, num_action_tokens: int, num_agents: int,
                 config: TransformerConfig, perattn_config: PerceiverConfig,
                 action_dim: int, use_bin: bool = False, bins: int = 64) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.use_bin = use_bin
        self.bins = bins

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
        act_tokens_pattern[-num_action_tokens:] = 1

        obs_tokens_pattern = torch.zeros(config.tokens_per_block)
        obs_tokens_pattern[:self.num_obs_tokens] = 1
        
        ### for autoregressive manner
        obs_autoregress_pattern = obs_tokens_pattern.clone()
        obs_autoregress_pattern = torch.roll(obs_autoregress_pattern, -1)

        ### due to attention mask, the last token of transformer output is generated by all tokens of input
        all_but_last_pattern = torch.zeros(config.tokens_per_block)
        all_but_last_pattern[-1] = 1

        ### Perceiver Attention output pattern
        perattn_pattern = torch.zeros(config.tokens_per_block)
        perattn_pattern[-num_action_tokens - 1 : -num_action_tokens] = 1

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
        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ELU(),
                nn.Linear(config.embed_dim, 1),
            )
        )
        
        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ELU(),
                nn.Linear(config.embed_dim, 2), # 这里改成了二元的termination预测
            )
        )

        self.action_dim = action_dim
        ## 注意这个avail_actions预测的是下一时刻的avail_actions
        self.heads_avail_actions = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ELU(),
                nn.Linear(config.embed_dim, action_dim),
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
        self.use_ib = True # use iris databuffer 

    def __repr__(self) -> str:
        return "multi_agent_world_model"

    def forward(self, tokens: torch.LongTensor, perattn_out: torch.Tensor = None, past_keys_values: Optional[KeysValues] = None) -> MAWorldModelOutput:
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

        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)

        # logits_last_action = self.head_last_action(x, num_steps=num_steps, prev_steps=prev_steps)
        # logits_last_action = rearrange(logits_last_action, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        pred_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        pred_rewards = rearrange(pred_rewards, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = rearrange(logits_ends, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        logits_avail_action = self.heads_avail_actions(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_avail_action = rearrange(logits_avail_action, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        return MAWorldModelOutput(x, logits_observations, pred_rewards, logits_ends, logits_avail_action)

    def compute_loss(self, batch, tokenizer: Tokenizer, **kwargs: Any):
        device = batch['observation'].device

        # only take discrete action space into account
        act_tokens = torch.argmax(batch['action'], dim=-1, keepdim=True)

        ### modified for ablation ###
        if not self.use_bin:
            with torch.no_grad():
                tokenizer_encodings = tokenizer.encode(batch['observation'], should_preprocess=True)  # (B, L, K)
                obs_tokens = tokenizer_encodings.tokens
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

        tokens = torch.cat([obs_tokens, torch.empty_like(act_tokens, device=device, dtype=torch.long), act_tokens], dim=-1) # (B, L, (K+N))
        tokens = rearrange(tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')  # (B, L(K+N))

        outputs = self(tokens, perattn_out = perattn_out)

        # compute labels
        if self.use_ib:  # if use iris databuffer
            valid_mask = batch['filled'].clone().unsqueeze(-1).expand(-1, -1, self.num_agents).to(torch.float32)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['reward'], batch['done'], batch['filled'])
            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b l o -> (b l) o')

            loss_obs = F.cross_entropy(logits_observations, labels_observations)

            pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs.logits_ends), 1)
            loss_ends = -(pred_ends.log_prob((1. - labels_ends)) * valid_mask).sum() / valid_mask.sum()

            l1_criterion = nn.SmoothL1Loss(reduction="none")
            loss_rewards = l1_criterion(outputs.pred_rewards, batch['reward'])
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
            pass

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
    
    def compute_labels_world_model_n(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def rollout_policy_trans(wm_env: MAWorldModelEnv, policy, horizons, initial_obs, initial_av_action):
    actor_feats = []
    critic_feats = []
    actions = []
    av_actions = []
    policies = []
    rewards = []
    dones = []

    # initialize wm_env
    rec_obs, critic_feat = wm_env.reset_from_initial_observations(initial_obs)

    av_action = initial_av_action
    for t in range(horizons):
        # feat = rearrange(wm_env.tokenizer.embedding(wm_env.obs_tokens), 'b n k e -> b n (k e)')
        # critic_feat = rearrange(wm_env.world_model.embedder.embedding_tables[1](wm_env.obs_tokens), 'b n k e -> b n (k e)')

        # action, pi = policy(feat)
        action, pi = policy(rec_obs)

        if av_action is not None:
            pi[av_action == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(av_action.squeeze(0))
        
        # actor_feats.append(feat)
        actor_feats.append(rec_obs)
        policies.append(pi)
        actions.append(action)
        critic_feats.append(critic_feat)

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
        
        
