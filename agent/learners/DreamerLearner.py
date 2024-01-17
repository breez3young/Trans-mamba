import sys
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict

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
from episode import SC2Episode
from dataset import MultiAgentEpisodesDataset

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
        # detect abnormal condition
        torch.autograd.set_detect_anomaly(True)

        self.config = config
        # self.model = DreamerModel(config).to(config.DEVICE).eval()
        # tokenizer
        self.tokenizer = Tokenizer(vocab_size=config.OBS_VOCAB_SIZE, embed_dim=config.EMBED_DIM,
                                   encoder=StateEncoder(config.encoder_config), decoder=StateDecoder(config.encoder_config)).to(config.DEVICE).eval()
        # ---------

        # world model (transformer)
        self.model = MAWorldModel(obs_vocab_size=config.OBS_VOCAB_SIZE, act_vocab_size=config.ACTION_SIZE, num_action_tokens=1, num_agents=config.NUM_AGENTS,
                                  config=config.trans_config, perceiver_config=config.perceiver_config, action_dim=config.ACTION_SIZE,
                                  is_continuous=False).to(config.DEVICE).eval()
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model).eval()
        # -------------------------
        # based on reconstructed obs
        self.actor = Actor(config.IN_DIM + config.TRANS_EMBED_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
        self.critic = AugmentedCritic(config.IN_DIM + config.TRANS_EMBED_DIM, config.HIDDEN).to(config.DEVICE)


        # initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        
        self.replay_buffer = MultiAgentEpisodesDataset(max_ram_usage="30G", name="train_dataset")
        # self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
        #                                    config.DEVICE, config.ENV_TYPE)

        self.entropy = config.ENTROPY
        self.step_count = -1
        self.train_count = 0
        self.cur_wandb_epoch = 0
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        self.tqdm_vis = True

    def init_optimizers(self):
        self.tokenizer_optimizer = torch.optim.Adam(self.tokenizer.parameters(), lr=self.config.t_lr)
        self.model_optimizer = configure_optimizer(self.model, self.config.wm_lr, self.config.wm_weight_decay)
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

        self.add_experience_to_dataset(rollout)
        # self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
        #                           rollout['fake'], rollout['last'], rollout.get('avail_action'))
        
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if self.replay_buffer.num_steps < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = self.accum_samples % self.config.N_SAMPLES
        sys.stdout.flush()

        self.train_count += 1

        intermediate_losses = defaultdict(float)
        # train tokenzier
        for i in tqdm(range(self.config.MODEL_EPOCHS), desc=f"Training {str(self.tokenizer)}", file=sys.stdout, disable=not self.tqdm_vis):
            samples = self.replay_buffer.sample_batch(batch_num_samples=self.config.t_bs,
                                                      sequence_length=1,
                                                      sample_from_start=True)
            samples = self._to_device(samples)
            loss_dict = self.train_tokenizer(samples)

            for loss_name, loss_value in loss_dict.items():
                intermediate_losses[loss_name] += loss_value / self.config.MODEL_EPOCHS

        utilization_rate = self.compute_utilization_rate()

        if self.train_count == 21:
            print('Start training world model...')
        if self.train_count > 20:
            # train transformer-based world model
            for i in tqdm(range(self.config.MODEL_EPOCHS), desc=f"Training {str(self.model)}", file=sys.stdout, disable=not self.tqdm_vis):
                samples = self.replay_buffer.sample_batch(batch_num_samples=self.config.MODEL_BATCH_SIZE,
                                                          sequence_length=self.config.SEQ_LENGTH,
                                                          sample_from_start=True)
                samples = self._to_device(samples)
                loss_dict = self.train_model(samples)

                for loss_name, loss_value in loss_dict.items():
                    intermediate_losses[loss_name] += loss_value / self.config.MODEL_EPOCHS

        if self.train_count == 46:
            print('Start training actor & critic...')
        if self.train_count > 45:
            # train actor-critic
            for i in tqdm(range(self.config.EPOCHS), desc=f"Training actor-critic", file=sys.stdout, disable=not self.tqdm_vis):
                samples = self.replay_buffer.sample_batch(batch_num_samples=self.config.BATCH_SIZE,
                                                          sequence_length=self.config.SEQ_LENGTH,
                                                          sample_from_start=False)
                samples = self._to_device(samples)
                self.train_agent_with_transformer(samples)

        wandb.log({'epoch': self.cur_wandb_epoch, **intermediate_losses})
        wandb.log({'epoch': self.cur_wandb_epoch, 'tokenizer/codebook_utilization_rate': utilization_rate})
        self.cur_wandb_epoch += 1
    
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

    @torch.no_grad()
    def compute_utilization_rate(self):
        cnt_tensor = torch.zeros(self.config.OBS_VOCAB_SIZE, device=self.config.DEVICE, dtype=torch.float32)
        for i in range(10):
            samples = self.replay_buffer.sample_batch(batch_num_samples=1024,
                                                      sequence_length=1,
                                                      sample_from_start=True)
            samples = self._to_device(samples)
            obs_tokens = self.tokenizer.encode(samples['observation']).tokens
            cur_cnt = torch.histc(obs_tokens, bins=self.config.OBS_VOCAB_SIZE, min=0, max=self.config.OBS_VOCAB_SIZE - 1)
            cnt_tensor += cur_cnt
        
        utilization_rate = (cnt_tensor != 0).sum() / self.config.OBS_VOCAB_SIZE
        return utilization_rate.item()

    def train_agent_with_transformer(self, samples):
        self.tokenizer.eval()
        self.model.eval()

        actions, av_actions, old_policy, feat, returns \
              = trans_actor_rollout(samples,
                                    self.tokenizer, self.model,
                                    self.actor,
                                    self.critic,
                                    self.config)
        
        adv = returns.detach() - self.critic(feat).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, feat[idx], returns[idx])
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

    ## add data to dataset
    def add_experience_to_dataset(self, data):
        episode = SC2Episode(
            observation=torch.FloatTensor(data['observation'].copy()),
            action=torch.FloatTensor(data['action'].copy()),
            av_action=torch.FloatTensor(data['avail_action'].copy()),
            reward=torch.FloatTensor(data['reward'].copy()),
            done=torch.FloatTensor(data['done'].copy()),
            filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
        )

        self.replay_buffer.add_episode(episode)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.config.DEVICE) for k in batch}

        