import torch.nn as nn
import torch.nn.functional as F

from networks.transformer.layers import AttentionEncoder
from networks.dreamer.utils import build_model


## based on joint local obs, predict the reward at the current timestep
class Reward_estimator(nn.Module):
    def __init__(self, in_dim, n_agents, hidden_size, activation=nn.ReLU):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim * n_agents)
        self.feedforward_model = build_model(in_dim * n_agents, 1, 3, hidden_size, activation)

    def forward(self, obss):
        obss = obss.reshape(*obss.shape[:-2], -1)
        obss = self.ln(obss)
        return self.feedforward_model(obss)