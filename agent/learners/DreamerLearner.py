import sys
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout, trans_actor_rollout
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import AugmentedCritic, Critic

from agent.models.tokenizer import Tokenizer, StateDecoder, StateEncoder
from agent.models.world_model import MAWorldModel
from utils import configure_optimizer

import wandb
import ipdb

def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:

    def __init__(self, config):
        self.config = config
        # self.model = DreamerModel(config).to(config.DEVICE).eval()
        # tokenizer
        self.encoder_config = config.encoder_config_fn(state_dim=config.IN_DIM)
        self.tokenizer = Tokenizer(vocab_size=config.OBS_VOCAB_SIZE, embed_dim=config.EMBED_DIM,
                                   encoder=StateEncoder(self.encoder_config), decoder=StateDecoder(self.encoder_config)).to(config.DEVICE).eval()
        # ---------

        # world model (transformer)
        self.model = MAWorldModel(obs_vocab_size=config.OBS_VOCAB_SIZE, act_vocab_size=config.ACTION_SIZE, num_action_tokens=1, num_agents=config.NUM_AGENTS,
                                  config=config.trans_config, perattn_config=config.perattn_config, action_dim=config.ACTION_SIZE,
                                  is_continuous=False).to(config.DEVICE).eval()
        # -------------------------

        self.actor = Actor(config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(
            config.DEVICE)
        self.critic = AugmentedCritic(config.critic_FEAT, config.HIDDEN).to(config.DEVICE)
        # self.critic = AugmentedCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)

        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_wandb_epoch = 0
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

    def init_optimizers(self):
        self.tokenizer_optimizer = torch.optim.Adam(self.tokenizer.parameters(), lr=self.config.t_lr, betas=(self.config.t_beta1, self.config.t_beta2))
        # self.model_optimizer = configure_optimizer(self.model, self.config.wm_lr, self.config.wm_weight_decay)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

    def params(self):
        return {'tokenizer': {k: v.cpu() for k, v in self.tokenizer.state_dict().items()},
                'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        intermediate_losses = defaultdict(float)
        for i in tqdm(range(self.config.MODEL_EPOCHS), desc=f"Training {str(self.tokenizer)}", file=sys.stdout, disable=True):
            samples = self.replay_buffer.sample_n(self.config.MODEL_BATCH_SIZE)
            loss_dict = self.train_tokenizer(samples)

            for loss_name, loss_value in loss_dict.items():
                intermediate_losses[loss_name] += loss_value / self.config.MODEL_EPOCHS


        for i in tqdm(range(self.config.MODEL_EPOCHS), desc=f"Training {str(self.model)}", file=sys.stdout, disable=True):
            samples = self.replay_buffer.sample_n(self.config.MODEL_BATCH_SIZE)
            loss_dict = self.train_model(samples)

            for loss_name, loss_value in loss_dict.items():
                intermediate_losses[loss_name] += loss_value / self.config.MODEL_EPOCHS

        for i in tqdm(range(self.config.EPOCHS), desc=f"Training actor-critic", file=sys.stdout, disable=True):
            samples = self.replay_buffer.sample_n(self.config.BATCH_SIZE)
            self.train_agent_with_transformer(samples)

        wandb.log({'epoch': self.cur_wandb_epoch, **intermediate_losses})
        self.cur_wandb_epoch += 1
        ### original code
        # for i in range(self.config.MODEL_EPOCHS):
        #     samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
        #     self.train_model(samples)

        

        # for i in range(self.config.EPOCHS):
        #     samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
        #     self.train_agent(samples)
    
    def train_tokenizer(self, samples):
        self.tokenizer.train()
        loss, loss_dict = self.tokenizer.compute_loss(samples)
        self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
        self.tokenizer.eval()
        return loss_dict
    
    def train_model(self, samples):
        self.model.train()
        loss, loss_dict = self.model.compute_loss(samples, self.tokenizer)
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.max_grad_norm) # or GRAD_CLIP
        self.model.eval()
        return loss_dict

    # def train_model(self, samples):
    #     self.model.train()
    #     loss = model_loss(self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
    #                       samples['reward'], samples['done'], samples['fake'], samples['last'])
    #     self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
    #     self.model.eval()

    def train_agent_with_transformer(self, samples):
        actions, av_actions, old_policy, actor_feat, critic_feat, returns \
              = trans_actor_rollout(samples['observation'],
                                    samples['av_action'],
                                    samples['last'],
                                    self.tokenizer, self.model,
                                    self.actor,
                                    self.critic,
                                    self.config)
        
        adv = returns.detach() - self.critic(critic_feat).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(actor_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, critic_feat[idx], returns[idx])
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

    def train_agent(self, samples):
        actions, av_actions, old_policy, imag_feat, returns = actor_rollout(samples['observation'],
                                                                            samples['action'],
                                                                            samples['last'], self.model,
                                                                            self.actor,
                                                                            self.critic if self.config.ENV_TYPE == Env.STARCRAFT
                                                                            else self.old_critic,
                                                                            self.config)
        adv = returns.detach() - self.critic(imag_feat).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, imag_feat[idx], returns[idx])
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
