from collections import OrderedDict, defaultdict
from pathlib import Path
import random
import shutil
from tqdm import tqdm

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


from agent.models.vq import SimpleVQAutoEncoder, SimpleFSQAutoEncoder
from agent.models.world_model import MAWorldModel
from agent.models.DreamerModel import DreamerModel
from networks.dreamer.action import Actor
from networks.dreamer.critic import AugmentedCritic, Critic

def load_mamba_model(config, ckpt_path):
    model = DreamerModel(config).eval()
    actor = Actor(config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).eval()
    critic = AugmentedCritic(config.FEAT, config.HIDDEN).eval()

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])

    return {
        "model": model.to(config.DEVICE),
        "actor": actor.to(config.DEVICE),
        "critic": critic.to(config.DEVICE),
    }

def load_mawm_model(config, ckpt_path):
    if config.tokenizer_type == 'vq':
        tokenizer = SimpleVQAutoEncoder(in_dim=config.IN_DIM, embed_dim=32, num_tokens=config.nums_obs_token,
                                        codebook_size=config.OBS_VOCAB_SIZE, learnable_codebook=False, ema_update=True).to(config.DEVICE).eval()
    elif config.tokenizer_type == 'fsq':
        levels = [8, 8, 8]
        tokenizer = SimpleFSQAutoEncoder(in_dim=config.IN_DIM, num_tokens=config.nums_obs_token, levels=levels).to(config.DEVICE).eval()

    if getattr(config, 'use_bin', None):
        obs_vocab_size = config.bins if config.use_bin else config.OBS_VOCAB_SIZE
    else:
        obs_vocab_size = config.OBS_VOCAB_SIZE

    perattn_config = config.perattn_config(num_latents=config.NUM_AGENTS)
    model = MAWorldModel(obs_vocab_size=obs_vocab_size, act_vocab_size=config.ACTION_SIZE, num_action_tokens=1, num_agents=config.NUM_AGENTS,
                         config=config.trans_config, perattn_config=perattn_config, action_dim=config.ACTION_SIZE,
                         use_bin=config.use_bin, bins=config.bins).to(config.DEVICE).eval()

    actor = Actor(config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
    critic = AugmentedCritic(config.critic_FEAT, config.HIDDEN).to(config.DEVICE)

    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)

    tokenizer.load_state_dict(ckpt['tokenizer'])
    model.load_state_dict(ckpt['model'])
    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])

    return {
        "tokenizer": tokenizer,
        "model": model,
        "actor": actor,
        "critic": critic,
    }
    
def _wrap(d):
    res = []
    for key, value in d.items():
        res.append(torch.tensor(value).float())
    
    res = torch.stack(res, dim=0)
    return res

@torch.no_grad()
def _compute_mawm_errors(model, sample, horizons):
    from agent.models.world_model_env import MAWorldModelEnv
    wm_env = MAWorldModelEnv(tokenizer=model["tokenizer"], world_model=model["model"], device=sample["observations"].device, env_name='sc2')
    
    gt_obs = sample["observations"]
    gt_actions = sample["actions"]
    gt_av_actions = sample["av_actions"]
    gt_r = sample["rewards"]
    
    pred_obs = []
    pred_r = []
    pred_av_actions = []
    pred_dis = []
    
    init_obs = gt_obs[0].unsqueeze(0)
    rec_obs, _ = wm_env.reset_from_initial_observations(init_obs)
    
    for t in range(horizons):
        pred_obs.append(rec_obs)
        
        rec_obs, reward, done, av_action, _ = wm_env.step(torch.argmax(gt_actions[t].unsqueeze(0), dim=-1).unsqueeze(-1), should_predict_next_obs=(t < horizons - 1))
        
        pred_av_actions.append(av_action)
        pred_r.append(reward)
        pred_dis.append(done)
        
    pred_obs = torch.concat(pred_obs, dim=0)
    pred_r = torch.concat(pred_r, dim=0)
    pred_av_actions = torch.concat(pred_av_actions, dim=0)
    pred_dis = torch.concat(pred_dis, dim=0)
    
    obs_l1_errors = (pred_obs - gt_obs).abs().sum(-1).mean()
    obs_l2_errors = (pred_obs - gt_obs).pow(2).sum(-1).mean()

    r_errors = (pred_r.squeeze() - gt_r).abs().mean()
    
    mean_dis = pred_dis.mean(0).squeeze()
    
    return {
        "obs_l1_errors": obs_l1_errors.item(),
        "obs_l2_errors": obs_l2_errors.item(),
        "r_errors": r_errors.item(),
        "mean_discount": mean_dis,
    }

from configs.dreamer.DreamerAgentConfig import RSSMState

def stack_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.stack)


def cat_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.cat)


def reduce_states(rssm_states: list, dim, func):
    return RSSMState(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                       for key in rssm_states[0].__dict__.keys()])

@torch.no_grad()
def _compute_mamba_errors(model_dict, sample, horizons):
    import ipdb
    from networks.dreamer.rnns import rollout_representation
    from agent.optim.loss import calculate_next_reward
    
    model = model_dict["model"]
    
    gt_obs = sample["observations"]
    gt_actions = sample["actions"]
    gt_av_actions = sample["av_actions"]
    gt_r = sample["rewards"]
    last = sample["last"]
    
    n_agents = gt_obs.shape[1]
    
    pred_av_actions = []
    
    embed = model.observation_encoder(gt_obs.reshape(-1, n_agents, gt_obs.shape[-1]))
    embed = embed.reshape(gt_obs.shape[0], 1, n_agents, -1)
    prev_state = model.representation.initial_state(1, n_agents, device=gt_obs.device)
    prior, post, _ = rollout_representation(model.representation, gt_obs.shape[0], embed, gt_actions[:-1].unsqueeze(1), prev_state, last.unsqueeze(-1).unsqueeze(1))
    
    post.stoch = post.stoch[0]
    post.deter = post.deter[0]
    post.logits = post.logits[0]
    state = post.map(lambda x: x.reshape(1, n_agents, -1))
    
    next_states = []
    tmp_next_states = []
    for t in range(horizons):
        next_states.append(state)
        tmp_next_states.append(state)
        state = model.transition(gt_actions[t + 1].unsqueeze(0), state)
    
    tmp_next_states.append(state)
    tmp_imag_states = cat_states(tmp_next_states, dim=0)
    tmp_imag_rew_feat = torch.cat([tmp_imag_states.stoch[:-1], tmp_imag_states.deter[1:]], -1)
    
    imag_states = cat_states(next_states, dim=0)
    imag_feat = imag_states.get_features()
    
    pred_r = calculate_next_reward(model, gt_actions[1:], imag_states)
    pred_obs = model.observation_decoder(imag_feat)[0]
    pred_dis = model.pcont(tmp_imag_rew_feat).mean
    
    obs_l1_errors = (pred_obs - gt_obs).abs().sum(-1).mean()
    obs_l2_errors = (pred_obs - gt_obs).pow(2).sum(-1).mean()

    r_errors = (pred_r.squeeze() - gt_r).abs().mean()
    
    mean_dis = pred_dis.mean(0).squeeze()
    
    return {
        "obs_l1_errors": obs_l1_errors.item(),
        "obs_l2_errors": obs_l2_errors.item(),
        "r_errors": r_errors.item(),
        "mean_discount": mean_dis,
    }

def compute_compounding_errors(models, sample, horizons):
    test_times = 10
    
    mawm_m, mamba_m = None, None
    for m in models:
        if "tokenizer" in m:
            mawm_m = m
        else:
            mamba_m = m
    
    length = sample["observations"].shape[0]
    
    c_mawm_errors = defaultdict(list)
    c_mamba_errors = defaultdict(list)
    
    for idx in range(test_times):
        print(f"--------------- Evaluation {idx}th time --------------")
        
        start = np.random.randint(1, length - horizons)
        end = start + horizons
        
        if mawm_m is not None:
            splitted_sample = {k: v[start:end] for k, v in sample.items()}
            error_dict = _compute_mawm_errors(mawm_m, splitted_sample, horizons)

            print(
                "Evaluating mawm - "
                + f"obs_l1_errors: {error_dict['obs_l1_errors']:.4f} | "
                + f"obs_l2_errors: {error_dict['obs_l2_errors']:.4f} | "
                + f"rew_l1_errors: {error_dict['r_errors']:.4f} | "
                + f"agent_aver_dis: {[round(v, 4) for v in error_dict['mean_discount'].tolist()]} | gt_ends?: {False if end != length else True}"
            )
            
            for k, v in error_dict.items():
                c_mawm_errors[k].append(v)
        
        if mamba_m is not None:
            splitted_sample = {
                "observations": sample["observations"][start:end],
                "actions": sample["actions"][start - 1 : end],
                "rewards": sample["rewards"][start:end],
                "av_actions": sample["av_actions"][start:end],
                "last": torch.zeros_like(sample["rewards"][start:end], device=sample["observations"].device)
            }
            
            error_dict = _compute_mamba_errors(mamba_m, splitted_sample, horizons)
            
            print(
                "Evaluating mamba - "
                + f"obs_l1_errors: {error_dict['obs_l1_errors']:.4f} | "
                + f"obs_l2_errors: {error_dict['obs_l2_errors']:.4f} | "
                + f"rew_l1_errors: {error_dict['r_errors']:.4f} | "
                + f"agent_aver_dis: {[round(v, 4) for v in error_dict['mean_discount'].tolist()]} | gt_ends?: {False if end != length else True}"
            )
        
            for k, v in error_dict.items():
                c_mamba_errors[k].append(v)

        print()
        
    print(
        f"Average {test_times} evaluations for MAWM: "
        + f"obs_l1_errors: {np.mean(c_mawm_errors['obs_l1_errors']):.4f} | "
        + f"obs_l2_errors: {np.mean(c_mawm_errors['obs_l2_errors']):.4f} | "
        + f"rew_l1_errors: {np.mean(c_mawm_errors['r_errors']):.4f}"
    )
    
    print(
        f"Average {test_times} evaluations for MAMBA: "
        + f"obs_l1_errors: {np.mean(c_mamba_errors['obs_l1_errors']):.4f} | "
        + f"obs_l2_errors: {np.mean(c_mamba_errors['obs_l2_errors']):.4f} | "
        + f"rew_l1_errors: {np.mean(c_mamba_errors['r_errors']):.4f}"
    )
    