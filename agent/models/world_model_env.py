import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
import torch.distributions as td
from torch.distributions.categorical import Categorical
import torchvision

from utils import action_split_into_bins

import ipdb


class MAWorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env_name: str, env: Optional[gym.Env] = None) -> None:
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env
        self.env_name = env_name
        ## necessary params
        self.mode = 'ar' if world_model.config.attention == 'causal' else 'bg'
        self.n_agents = world_model.num_agents
        self.predict_avail_action = True if world_model.action_dim else False

        self.is_continuous = world_model.is_continuous

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    ## unmodified
    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, N, Obs_dim) -> (B, N, K)
        num_observations_tokens = obs_tokens.shape[-1]
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        output_sequence = self.refresh_keys_values_with_initial_obs_tokens(rearrange(obs_tokens, 'b n k -> (b n) k'))
        self.obs_tokens = obs_tokens
        # trans_feat = rearrange(output_sequence[:, -1], '(b n) k -> b n k', b=int(output_sequence.size(0) / self.n_agents), n=self.n_agents)
        return self.decode_obs_tokens() #, trans_feat

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape # (B, K)
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.Tensor], should_predict_next_obs: bool = True):
        if type(action) == np.ndarray:
            action = torch.tensor(action, device=self.device)

        if self.mode == 'ar':
            return self.step_ar(action=action, should_predict_next_obs=should_predict_next_obs)
        elif self.mode == 'bg':
            return self.step_bg(action=action, should_predict_next_obs=should_predict_next_obs)
        else:
            raise ValueError(f'Mode {self.mode} has no corresponding step!')

    ### TODO
    @torch.no_grad()
    def step_bg(self, action: Union[List, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True):
        raise NotImplementedError

    # 
    @torch.no_grad()
    def step_ar(self, action: Union[List, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True):
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)
        
        if self.is_continuous:
            action = action_split_into_bins(action, self.world_model.act_vocab_size)

        # perceiver attention output
        perattn_out = self.world_model.get_perceiver_out(self.obs_tokens, action)
        extra_info = {'perattn_out': perattn_out.detach().clone() if should_predict_next_obs else None}
        perattn_out = rearrange(perattn_out, 'b n e -> (b n) 1 e')
        # ---------------------------

        token = torch.cat([torch.zeros_like(action, dtype=torch.long, device=action.device), action], dim=-1)
        token = rearrange(token, 'b n k -> (b n) k').to(self.device)  # (B, N)

        for k in range(num_passes):  # assumption that there is only one action token.
            outputs_wm = self.world_model(token, perattn_out=perattn_out, past_keys_values=self.keys_values_wm)
            ### 当head拿不到对应的slice indices，某些head的输出就是e.g. tensor([], size=(64, 0, 2))
            output_sequence.append(outputs_wm.output_sequence)
            
            if k == 0:
                reward = outputs_wm.pred_rewards.float()
                
                # done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool)  # (B,), numpy
                pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs_wm.logits_ends), 1)
                done = pred_ends.mean

                if self.predict_avail_action:
                    avail_action_dist = td.independent.Independent(td.Bernoulli(logits=outputs_wm.pred_avail_action), 1)
                    avail_action = avail_action_dist.sample()
                else:
                    avail_action = None

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)
            
            perattn_out = None

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        obs_tokens = torch.cat(obs_tokens, dim=1)             # (B, K)

        self.obs_tokens = rearrange(obs_tokens, '(b n) k -> b n k', b=int(obs_tokens.size(0) / self.n_agents), n=self.n_agents)

        reward = reward.squeeze(1)
        done = done.squeeze(1)
        avail_action = avail_action.squeeze(1) if avail_action is not None else None
        obs = self.decode_obs_tokens() if should_predict_next_obs else None # obs is tensor
        extra_info['obs_tokens'] = self.obs_tokens.detach().clone() if should_predict_next_obs else None

        return obs, reward, done, avail_action, extra_info

    ## unmodified
    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self):
        # assert self.obs_tokens.shape[0] % self.n_agents == 0
        # bs = self.obs_tokens.shape[0]
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        rec = self.tokenizer.decode(embedded_tokens, should_postprocess=True)
        # rec = rearrange(rec, '(b n) o -> b n o', b=int(bs / self.n_agents), n=self.n_agents)
        if self.env_name == "sc2":
            return torch.clamp(rec, -1., 1.)
        elif self.env_name == "maniskill2":
            return rec
        else:
            raise ValueError(f'Unsupported env {self.env_name}')

    ## unmodified
    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]