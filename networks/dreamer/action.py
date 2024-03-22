from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.distributions import OneHotCategorical
from networks.transformer.layers import AttentionEncoder
from networks.dreamer.utils import build_model


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()

        self.feedforward_model = build_model(in_dim, out_dim, layers, hidden_size, activation)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x


class AttentionActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, out_dim, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)

    def forward(self, state_features):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        x = self.feedforward_model(attn_embeds)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x


class RNNActor(nn.Module):
    def __init__(self, in_dim, out_dim, n_agents, hidden_size = 512, layers = 3, use_original_obs: bool = False, activation=nn.ReLU) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.n_agents = n_agents

        self.embed = build_model(in_dim, hidden_size, layers, activation=activation)

        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        self.actor_linear = nn.Linear(512, out_dim)

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        device = burnin_observations.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 3 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None):
        assert inputs.ndim == 2  # input shape: (batch size * n_agents, input_dim)
        assert -1 <= inputs.min() <= 1 and -1 <= inputs.max() <= 1
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        x = inputs[mask_padding] if mask_padding is not None else inputs

        b, e = x.shape
        # x = x.mul(2).sub(1)
        x = self.embed(x)

        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        logits_actions = self.actor_linear(self.hx)  # rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')

        return logits_actions