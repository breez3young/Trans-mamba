from collections import OrderedDict
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn


def action_split_into_bins(actions, bins: int):
    # assume space of actions should be Box(-1, 1)
    EPS = 1e-10
    boundaries = torch.linspace(-1 - EPS, 1, bins + 1, device=actions.device, dtype=torch.float64)
    bucketized_act = torch.bucketize(actions.contiguous(), boundaries) - 1
    return bucketized_act.to(actions.device)

def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.MultiheadAttention)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if pn == "perattn.latents":
                no_decay.add(pn)
            
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def joint2localActions(actions, avail_actions):
    ### used for maniskill2
    import pdb
    assert avail_actions.sum() == actions.shape[-1]
    local_actions = torch.zeros_like(avail_actions, dtype=torch.float32, device=avail_actions.device)
    s = 0
    for idx in range(avail_actions.size(0)):
        length = avail_actions[idx].sum().item()
        local_actions[idx][avail_actions[idx] == 1] = torch.tensor(actions[s : s + length], device=avail_actions.device)
        s += length
    
    return local_actions


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

## 以下两个函数都默认源域是[-1., 1.]
## discretize
@torch.no_grad()
def discretize_into_bins(obs, bins: int):
    eps = 1e-6
    boundaries = torch.linspace(-1 - eps, 1, bins + 1, device=obs.device, dtype=torch.float32)
    obs_tokens = torch.bucketize(obs, boundaries) - 1
    return obs_tokens.to(obs.device)

@torch.no_grad()
def bins2continuous(obs_tokens, bins: int):
    boundaries = torch.linspace(-1, 1, bins + 1, device=obs_tokens.device, dtype=torch.float32)
    numerical_map = (boundaries[:-1] + boundaries[1:]) / 2
    return numerical_map[obs_tokens]
    
    
def action_split_into_bins(actions, bins: int):
    # assume space of actions should be Box(-1, 1)
    eps = 1e-6
    boundaries = torch.linspace(-1 - eps, 1, bins + 1, device=actions.device, dtype=torch.float32)
    bucketized_act = torch.bucketize(actions.contiguous(), boundaries) - 1
    return bucketized_act.to(actions.device)

def generate_group_name(args, config):
    if getattr(config, 'use_bin', None):
        use_vq = True
    else:
        use_vq = not config.use_bin
    
    if not use_vq:
        g_name = f'{args.env_name}_H{config.HORIZON}_X{config.bins}'
    else:
        g_name = f'{args.env_name}_H{config.HORIZON}_T{config.nums_obs_token}_Vocab{config.OBS_VOCAB_SIZE}_{args.tokenizer}'
    
    return g_name
