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
from .transformer import Transformer, TransformerConfig, PerAttention, PerceiverConfig, get_sinusoid_encoding_table, PerBlock, Perceiver
from .world_model_env import MAWorldModelEnv
from utils import init_weights, LossWithIntermediateLosses, action_split_into_bins
import wandb
import ipdb

@dataclass
class MAWorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    pred_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    pred_avail_action: torch.FloatTensor
    logits_last_action: torch.FloatTensor


class MAWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, num_action_tokens: int, num_agents: int,
                 config: TransformerConfig, perceiver_config: PerceiverConfig,
                 action_dim: int, is_continuous: bool = False) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.is_continuous = is_continuous

        self.config = config
        self.num_agents = num_agents

        ## perceiver attention
        self.perceiver_config = perceiver_config
        self.perceiver = Perceiver(**dataclasses.asdict(perceiver_config))
        # self.bidirectional_attn_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=perattn_config.context_dim, nhead=perattn_config.heads,
        #                                                                                    dim_feedforward=perattn_config.context_dim,
        #                                                                                    dropout=perattn_config.dropout, activation='gelu',
        #                                                                                    batch_first=True, norm_first=True), 2)
        # self.before_q = nn.Parameter(torch.randn(num_agents, perattn_config.query_dim))

        self.agent_id_pos_emb = get_sinusoid_encoding_table(30, perceiver_config.dim)
        self.agent_id_pos_emb.requires_grad_(False)
        ## --------------------

        self.num_action_tokens = num_action_tokens  # for continuous task, this should be dimension of joint action (e.g. like ManiSkill2)
        self.num_obs_tokens = config.tokens_per_block - num_action_tokens - 1  # 其中有一个是perceiver attn的输出

        self.transformer = Transformer(config)

        act_tokens_pattern = torch.zeros(config.tokens_per_block)
        # act_tokens_pattern[-num_action_tokens - 1:-1] = 1
        act_tokens_pattern[-1] = 1

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
        perattn_pattern[-num_action_tokens - 1:-1] = 1

        self.perattn_slicer = Slicer(max_blocks=config.max_blocks, block_mask=perattn_pattern)

        # self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

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
                nn.Linear(config.embed_dim, 1),
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

    def __repr__(self) -> str:
        return "multi_agent_world_model"

    def forward(self, tokens: torch.LongTensor, perattn_out: torch.Tensor = None, past_keys_values: Optional[KeysValues] = None, tokens_mask: Optional[torch.Tensor] = None) -> MAWorldModelOutput:
        bs = tokens.size(0)
        num_steps = tokens.size(1)  # (B, T)

        assert num_steps <= self.config.max_tokens
        assert tokens_mask is None or tokens_mask.shape == tokens.shape[:2]
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        
        sequences = self.embedder(tokens, num_steps, prev_steps)

        indices = self.perattn_slicer.compute_slice(num_steps, prev_steps)

        if perattn_out is not None:
            assert len(indices) != 0
            sequences[:, indices] = perattn_out
        else:
            assert len(indices) == 0

        x = self.transformer(sequences, past_keys_values = past_keys_values, input_mask = tokens_mask)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)

        # logits_last_action = self.head_last_action(x, num_steps=num_steps, prev_steps=prev_steps)
        # logits_last_action = rearrange(logits_last_action, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        pred_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        pred_rewards = rearrange(pred_rewards, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = rearrange(logits_ends, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        logits_avail_action = self.heads_avail_actions(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_avail_action = rearrange(logits_avail_action, '(b n) l e -> b l n e', b=int(bs / self.num_agents), n=self.num_agents)

        return MAWorldModelOutput(x, logits_observations, pred_rewards, logits_ends, logits_avail_action, None)

    def compute_loss(self, batch, tokenizer: Tokenizer, **kwargs: Any):
        device = batch['observation'].device
        if not self.is_continuous:
            act_tokens = torch.argmax(batch['action'], dim=-1, keepdim=True)

        with torch.no_grad():
            tokenizer_encodings = tokenizer.encode(batch['observation'], should_preprocess=True)  # (B, L, K)
            obs_tokens = tokenizer_encodings.tokens

        ### 将obs encodings和action encodings一起过perceiver attention
        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](act_tokens)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        b, l, N, M, e = input_encodings.shape

        agent_id_emb = repeat(self.agent_id_pos_emb[:, :self.num_agents], '1 n e -> (b l) (n m) e', b = b, l = l, m = M)
        input_encodings = rearrange(input_encodings, 'b l n m e -> (b l) (n m) e') + agent_id_emb.detach().to(device)

        perattn_out = self.perceiver(input_encodings)
        
        perattn_out = rearrange(perattn_out, '(b l) n e -> (b n) l e', b=b, l=l, n=N)  ### TODO 不知道这样直接调换dim会不会和transpose有问题
        
        if not self.is_continuous:
            # tokens = torch.cat([obs_tokens, torch.zeros(b, l, N, 1, dtype=torch.long).to(device), act_tokens], dim=-1)
            tokens = torch.cat([obs_tokens, torch.empty(b, l, N, 1, dtype=torch.long).to(device), act_tokens], dim=-1)
        else:
            act_tokens = action_split_into_bins(batch['joint_actions'], self.act_vocab_size) if 'joint_actions' in batch else action_split_into_bins(batch['action'], self.act_vocab_size)
            tokens = torch.cat([obs_tokens, torch.zeros(b, l, N, 1, dtype=torch.long).to(device), act_tokens], dim=-1)

        ### 生成perattn的占位符
        # tokens = torch.cat([tokens, torch.zeros(b, l, N, 1, dtype=torch.long).clone().detach().to(device)], dim=-1)  # 用于对perceiver attention output的占位
        tokens = rearrange(tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')  # (B, L(K+N))
        # tokens_mask = repeat(batch['filled'], 'b l -> (b n) (l k)', n = N, k = self.config.tokens_per_block)
        outputs = self(tokens, perattn_out=perattn_out, tokens_mask=None)

        # compute labels
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['reward'], batch['done'], batch['filled'])
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b l o -> (b l) o')

        loss_obs = F.cross_entropy(logits_observations, labels_observations)

        valid_mask = batch['filled'].clone().unsqueeze(-1).expand(-1, -1, self.num_agents).to(torch.float32)
        # loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
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

        pred_av_actions = td.independent.Independent(td.Bernoulli(logits=outputs.pred_avail_action), 1)
        loss_av_actions = -(pred_av_actions.log_prob(batch['av_action']) * valid_mask).sum() / valid_mask.sum()

        w1 = 2.0
        w2 = 2.0
        w3 = 1.0

        loss = loss_obs + loss_rewards + loss_ends + loss_av_actions + info_loss

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
        labels_ends = ends.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(ends), 1.) 
        
        return labels_observations.reshape(-1), labels_rewards, labels_ends
    
    def compute_labels_world_model_n(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        labels_observations = rearrange(obs_tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')[:, 1:]
        labels_rewards = rearrange(rewards.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        labels_ends = rearrange(ends.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1).to(torch.long)
    
    
    def get_perceiver_out(self, obs_tokens, act_tokens):
        device = obs_tokens.device
        shape = obs_tokens.shape
        
        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](act_tokens)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        n, m, e = input_encodings.shape[-3:]
        input_encodings = rearrange(input_encodings, '... n m e -> (...) (n m) e')
        agent_id_emb = repeat(self.agent_id_pos_emb[:, :n], '1 n e -> b (n m) e', b = input_encodings.size(0), n = n, m = m)

        input_encodings += agent_id_emb.detach().to(device)
        perattn_out = self.perceiver(input_encodings)

        perattn_out = perattn_out.reshape(*shape[:-1], -1)
        return perattn_out


def rollout_policy_trans(tokenizer, world_model, policy, config, batch):
    # credits to https://github.com/pytorch/pytorch/issues/64208
    ## helper
    gather_incomplete_left = lambda tensor, I: tensor.gather(I.ndim, I[(...,) + (None,) * (tensor.ndim - I.ndim)].expand((-1,) * (I.ndim + 1) + tensor.shape[I.ndim + 1:])).squeeze(I.ndim)
    ## ------

    bs = batch['observation'].size(0)
    n_agents = batch['observation'].size(2)
    device = batch['observation'].device
    horizons = config.HORIZON
    tokens_per_timestep = world_model.config.tokens_per_block

    wm_env = MAWorldModelEnv(tokenizer=tokenizer, world_model=world_model, device=config.DEVICE, env_name='sc2')

    # compute history info for the first time
    history_obs_tokens = tokenizer.encode(batch['observation'].clone(), should_preprocess=True).tokens
    history_act_tokens = torch.argmax(batch['action'], dim=-1, keepdim=True)
    history_perattn_out = world_model.get_perceiver_out(history_obs_tokens, history_act_tokens)
    history_tokens = torch.cat([history_obs_tokens, torch.zeros_like(history_act_tokens, dtype=torch.long).to(device), history_act_tokens], dim=-1)

    del history_obs_tokens
    del history_act_tokens

    history_filled = batch['filled'].clone()
    # ipdb.set_trace()
    outputs_sequence = world_model(rearrange(history_tokens, 'b l n k -> (b n) (l k)'),
                                   perattn_out=rearrange(history_perattn_out, 'b l n e -> (b n) l e'),
                                   tokens_mask=repeat(history_filled, 'b l -> (b n) (l k)', n = n_agents, k = tokens_per_timestep)).output_sequence

    trans_feat = rearrange(outputs_sequence[:, -4], '(b n) e -> b n e', b = bs, n = n_agents) # -3

    feats = []
    actions = []
    av_actions = []
    policies = []
    rewards = []
    dones = []

    # initialize wm_env
    initial_obs = batch['observation'][:, -1]
    av_action = batch['av_action'][:, -1]

    rec_obs = wm_env.reset_from_initial_observations(initial_obs)
    for t in range(horizons):
        feat = torch.cat([rec_obs, trans_feat], dim=-1)
        action, pi = policy(feat)

        if av_action is not None:
            pi[av_action == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(av_action.squeeze(0))
        
        # actor_feats.append(feat)
        # critic_feats.append(feat)
        feats.append(feat)
        policies.append(pi)
        actions.append(action)

        rec_obs, reward, done, av_action, extra_info = wm_env.step(torch.argmax(action, dim=-1).unsqueeze(-1), should_predict_next_obs=(t < horizons - 1))

        # update history info
        if t < horizons - 1:
            cur_tokens = torch.cat([extra_info['obs_tokens'], torch.zeros(bs, n_agents, 1, dtype = torch.long ,device = device), torch.argmax(action, dim=-1, keepdim=True)], dim=-1)
            cur_perattn_out = extra_info['perattn_out']
            history_tokens = torch.roll(history_tokens, -1, dims=1)
            history_perattn_out = torch.roll(history_perattn_out, -1, dims=1)
            history_filled = torch.roll(history_filled, -1, dims=1)
            history_tokens[:, -1] = cur_tokens
            history_perattn_out[:, -1] = cur_perattn_out
            history_filled[:, -1] = True

            outputs_sequence = world_model(rearrange(history_tokens, 'b l n k -> (b n) (l k)'),
                                        perattn_out=rearrange(history_perattn_out, 'b l n e -> (b n) l e'),
                                        tokens_mask=repeat(history_filled, 'b l -> (b n) (l k)', n = n_agents, k = tokens_per_timestep)).output_sequence
            
            trans_feat = rearrange(outputs_sequence[:, -4], '(b n) e -> b n e', b = bs, n = n_agents) # -3
        # --------------------- dividing line -----------------------
        
        rewards.append(reward)
        dones.append(done)

    return {"feats": torch.stack(feats, dim=0),
            #"actor_feats": torch.stack(actor_feats, dim=0),
            #"critic_feats": torch.stack(critic_feats, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "discounts": torch.stack(dones, dim=0),
            }
        
        
