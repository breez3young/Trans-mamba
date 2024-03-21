import sys
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict
from einops import rearrange
from torch.utils.data.dataloader import DataLoader

import numpy as np
import torch

from agent.memory.DreamerMemory import DreamerMemory, ObsDataset
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout, trans_actor_rollout
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import AugmentedCritic, Critic

from agent.models.tokenizer import Tokenizer, StateDecoder, StateEncoder
from agent.models.vq import SimpleVQAutoEncoder, SimpleFSQAutoEncoder
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
        self.config = config
        self.config.update()

        # tokenizer
        # self.encoder_config = config.encoder_config_fn(state_dim=config.IN_DIM)
        # self.tokenizer = Tokenizer(vocab_size=config.OBS_VOCAB_SIZE, embed_dim=config.EMBED_DIM,
        #                            encoder=StateEncoder(self.encoder_config), decoder=StateDecoder(self.encoder_config)).to(config.DEVICE).eval()
        if self.config.tokenizer_type == 'vq':
            self.tokenizer = SimpleVQAutoEncoder(in_dim=config.IN_DIM, embed_dim=32, num_tokens=config.nums_obs_token,
                                                 codebook_size=config.OBS_VOCAB_SIZE, learnable_codebook=False, ema_update=True, decay=config.ema_decay).to(config.DEVICE).eval()
            self.obs_vocab_size = config.OBS_VOCAB_SIZE
        elif self.config.tokenizer_type == 'fsq':
            # 2^8 -> [8, 6, 5], 2^10 -> [8, 5, 5, 5]
            levels = [8, 8, 8]
            self.tokenizer = SimpleFSQAutoEncoder(in_dim=config.IN_DIM, num_tokens=config.nums_obs_token, levels=levels).to(config.DEVICE).eval()
            self.obs_vocab_size = np.prod(levels)
        else:
            raise NotImplementedError
        # ---------

        # world model (transformer)
        obs_vocab_size = config.bins if config.use_bin else config.OBS_VOCAB_SIZE
        perattn_config = config.perattn_config(num_latents=config.NUM_AGENTS)
        self.model = MAWorldModel(obs_vocab_size=obs_vocab_size, act_vocab_size=config.ACTION_SIZE, num_action_tokens=1, num_agents=config.NUM_AGENTS,
                                  config=config.trans_config, perattn_config=perattn_config, action_dim=config.ACTION_SIZE,
                                  use_bin=config.use_bin, bins=config.bins, use_classification=False).to(config.DEVICE).eval()
        # -------------------------

        # based on latent
        # self.actor = Actor(config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
        # self.critic = AugmentedCritic(config.critic_FEAT, config.HIDDEN).to(config.DEVICE)

        # based on reconstructed obs
        if not self.config.use_stack:
            self.actor = Actor(config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
            self.critic = AugmentedCritic(config.IN_DIM, config.HIDDEN).to(config.DEVICE)
        
        else:
            print(f"Use stacking observation mode. Currently stack {config.stack_obs_num} observations for decision making.")
            self.actor = Actor(config.IN_DIM * config.stack_obs_num, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
            self.critic = AugmentedCritic(config.IN_DIM * config.stack_obs_num, config.HIDDEN).to(config.DEVICE)

        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        
        self.replay_buffer = MultiAgentEpisodesDataset(max_ram_usage="30G", name="train_dataset", temp=20)
        
        ## (debug) pre-load mamba training buffer
        if self.config.is_preload:
            print(f"Load offline dataset from {self.config.load_path}")
            self.replay_buffer.load_from_pkl(self.config.load_path)
        
        if self.config.use_external_rew_model:
            from networks.dreamer.reward_estimator import Reward_estimator

            pretrained_rew_model_path = "/mnt/data/optimal/zhangyang/code/bins/pretrained_weights/ckpt/2024-03-15_13-53-32-722/rew_model_ep113.pth"
            print(f"Load pretrained reward model from {pretrained_rew_model_path}")

            self.rew_model = Reward_estimator(in_dim=config.IN_DIM + config.ACTION_SIZE, hidden_size=256, n_agents=config.NUM_AGENTS).to(config.DEVICE)
            checkpoint = torch.load(pretrained_rew_model_path)
            self.rew_model.load_state_dict(checkpoint['model'])
            self.rew_model.eval()
        
        ### world model dataset
        self.mamba_replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                                 config.DEVICE, config.ENV_TYPE, config.sample_temperature)
        
        ### tokenizer dataset
        # self.vq_buffer = ObsDataset(config.CAPACITY, config.IN_DIM, config.NUM_AGENTS)

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

        if self.config.use_bin:
            print('Not using & training tokenizer...')

        if self.config.critic_average_r:
            print("Enable average mode for predicted rewards...")
        else:
            print("Disable average mode for predicted rewards...")

    def init_optimizers(self):
        # self.tokenizer_optimizer = torch.optim.Adam(self.tokenizer.parameters(), lr=self.config.t_lr)
        self.tokenizer_optimizer = torch.optim.AdamW(self.tokenizer.parameters(), lr=3e-4)
        self.model_optimizer = configure_optimizer(self.model, self.config.wm_lr, self.config.wm_weight_decay)
        # self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

    def params(self):
        return {'tokenizer': {k: v.cpu() for k, v in self.tokenizer.state_dict().items()},
                'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}
        
    def load_pretrained_wm(self, load_path):
        ckpt = torch.load(load_path)
        self.tokenizer.load_state_dict(ckpt['tokenizer'])
        self.model.load_state_dict(ckpt['model'])
        
        self.tokenizer.eval()
        self.model.eval()

    def save(self, save_path):
        torch.save(self.params(), save_path)

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])

        self.add_experience_to_dataset(rollout)
        self.mamba_replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                        rollout['fake'], rollout['last'], rollout.get('avail_action'))
        # self.vq_buffer.append(rollout['observation'])
        
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if self.replay_buffer.num_steps < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        self.train_count += 1

        intermediate_losses = defaultdict(float)
        # train tokenzier
        if not self.config.use_bin:
            # vq_loader = DataLoader(
            #     self.vq_buffer,
            #     shuffle=True,
            #     pin_memory=True,
            #     batch_size=64,
            #     num_workers=1,
            # )
            # pbar = tqdm(enumerate(vq_loader), total=len(vq_loader), desc=f"Training tokenizer", file=sys.stdout, disable=not self.tqdm_vis)
            pbar = tqdm(range(self.config.WM_EPOCHS), desc=f"Training tokenizer", file=sys.stdout, disable=not self.tqdm_vis)
            for _ in pbar:
                samples = self.mamba_replay_buffer.sample_batch(bs=256, sl=1, mode="tokenizer")
                samples = self._to_device(samples)

                # loss_dict = self.train_tokenizer(samples)
                if self.config.tokenizer_type == 'vq':
                    loss_dict = self.train_vq_tokenizer(samples['observation'])

                    pbar.set_description(
                        f"Training tokenizer:"
                        + f"rec loss: {loss_dict[self.config.tokenizer_type + '/rec_loss']:.4f} | "
                        + f"cmt loss: {loss_dict[self.config.tokenizer_type + '/cmt_loss']:.4f} | "
                        + f"active %: {loss_dict[self.config.tokenizer_type + '/active']:.3f} | "
                    )
                elif self.config.tokenizer_type == 'fsq':
                    loss_dict = self.train_fsq_tokenizer(samples['observation'])

                    pbar.set_description(
                        f"Training tokenizer:"
                        + f"rec loss: {loss_dict[self.config.tokenizer_type + '/rec_loss']:.4f} | "
                        + f"active %: {loss_dict[self.config.tokenizer_type + '/active']:.3f} | "
                    )
                else:
                    raise NotImplementedError

                for loss_name, loss_value in loss_dict.items():
                    intermediate_losses[loss_name] += loss_value / self.config.MODEL_EPOCHS

        if self.train_count == 15:
            print('Start training world model...')

        if self.train_count > 14:
            # train transformer-based world model
            pbar = tqdm(range(self.config.WM_EPOCHS), desc=f"Training {str(self.model)}", file=sys.stdout, disable=not self.tqdm_vis)
            for _ in pbar:
                samples = self.mamba_replay_buffer.sample_batch(bs=self.config.MODEL_BATCH_SIZE, sl=self.config.SEQ_LENGTH, mode="model")
                samples = self._to_device(samples)
                attn_mask = self.mamba_replay_buffer.generate_attn_mask(samples["done"], self.model.config.tokens_per_block).to(self.config.DEVICE)

                loss_dict = self.train_model(samples, attn_mask)

                for loss_name, loss_value in loss_dict.items():
                    intermediate_losses[loss_name] += loss_value / self.config.MODEL_EPOCHS

                pbar.set_description(
                    f"Training world model:"
                    + f"total loss: {loss_dict['world_model/total_loss']:.4f} | "
                    + f"obs loss: {loss_dict['world_model/loss_obs']:.4f} | "
                    + f"rew loss: {loss_dict['world_model/loss_rewards']:.4f} | "
                    + f"dis loss: {loss_dict['world_model/loss_ends']:.3f} | "
                    + f"av loss: {loss_dict['world_model/loss_av_actions']:.3f} | "
                )

        if self.train_count == 40:
            print('Start training actor & critic...')

        if self.train_count > 39:
            # train actor-critic
            for i in tqdm(range(self.config.EPOCHS), desc=f"Training actor-critic", file=sys.stdout, disable=not self.tqdm_vis):
                samples = self.replay_buffer.sample_batch(batch_num_samples=600, # self.config.MODEL_BATCH_SIZE * 2
                                                          sequence_length=self.config.stack_obs_num if self.config.use_stack else 1,
                                                          sample_from_start=False,
                                                          valid_sample=False)
                
                # samples = self.mamba_replay_buffer.sample_batch(bs=30, sl=20)

                samples = self._to_device(samples)
                self.train_agent_with_transformer(samples)

        wandb.log({'epoch': self.cur_wandb_epoch, **intermediate_losses})
        
        if self.train_count % 200 == 0 and self.train_count > 19:
            self.model.eval()
            self.tokenizer.eval()
            sample = self.replay_buffer.sample_batch(batch_num_samples=1,
                                                     sequence_length=self.config.HORIZON,
                                                     sample_from_start=True,
                                                     valid_sample=True)
            sample = self._to_device(sample)
            self.model.visualize_attn(sample, self.tokenizer, Path(self.config.RUN_DIR) / "visualization" / "attn" / f"epoch_{self.train_count}")
        
        self.cur_wandb_epoch += 1
    
    def train_vq_tokenizer(self, x):
        assert type(self.tokenizer) == SimpleVQAutoEncoder
        self.tokenizer.train()

        out, indices, cmt_loss = self.tokenizer(x, True, True)
        rec_loss = (out - x).abs().mean()
        loss = rec_loss + self.config.alpha * cmt_loss

        active_rate = indices.detach().unique().numel() / self.obs_vocab_size * 100

        self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
        self.tokenizer.eval()

        loss_dict = {
            self.config.tokenizer_type + "/cmt_loss": cmt_loss.item(),
            self.config.tokenizer_type + "/rec_loss": rec_loss.item(),
            self.config.tokenizer_type + "/active": active_rate,
        }

        return loss_dict

    def train_fsq_tokenizer(self, x):
        assert type(self.tokenizer) == SimpleFSQAutoEncoder
        self.tokenizer.train()

        out, indices = self.tokenizer(x, True, True)
        loss = (out - x).abs().mean()

        active_rate = indices.detach().unique().numel() / self.obs_vocab_size * 100

        self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
        self.tokenizer.eval()

        loss_dict = {
            self.config.tokenizer_type + "/rec_loss": loss.item(),
            self.config.tokenizer_type + "/active": active_rate,
        }

        return loss_dict

    # def train_tokenizer(self, samples):
    #     self.tokenizer.train()
    #     loss, loss_dict = self.tokenizer.compute_loss(samples)
    #     self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
    #     self.tokenizer.eval()
    #     return loss_dict
    
    def train_model(self, samples, attn_mask = None):
        self.tokenizer.eval()
        self.model.train()
        
        # loss, loss_dict = self.model.compute_loss(samples, self.tokenizer)
        loss, loss_dict = self.model.compute_loss(samples, self.tokenizer, attn_mask)
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.max_grad_norm) # or GRAD_CLIP
        self.model.eval()
        return loss_dict

    def train_agent_with_transformer(self, samples):
        self.tokenizer.eval()
        self.model.eval()

        actions, av_actions, old_policy, actor_feat, critic_feat, returns \
              = trans_actor_rollout(samples['observation'],  # rearrange(samples['observation'], 'b l n e -> (b l) 1 n e'),
                                    samples['av_action'],  # rearrange(samples['av_action'], 'b l n e -> (b l) 1 n e'),
                                    samples['filled'], # samples['last']
                                    self.tokenizer, self.model,
                                    self.actor,
                                    self.critic, # self.critic
                                    self.config,
                                    external_rew_model=self.rew_model if self.config.use_external_rew_model else None,
                                    use_stack=self.config.use_stack,
                                    stack_obs_num=self.config.stack_obs_num if self.config.use_stack else None,
                                    )
        
        adv = returns.detach() - self.critic(critic_feat).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        
        self.cur_update += 1
        
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                idx = inds[i:i + step]
                loss = actor_loss(actor_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy)
                actor_grad_norm = self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, critic_feat[idx], returns[idx])
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                critic_grad_norm = self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                
                wandb.log({'Agent/actor_grad_norm': actor_grad_norm, 'Agent/critic_grad_norm': critic_grad_norm})
        
        # hard update critic
        if self.cur_update % self.config.TARGET_UPDATE == 0:
            self.old_critic = deepcopy(self.critic)
            self.cur_update = 0

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        return grad_norm

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

    
    ### for offline pretraining world model
    def train_wm_offline(self, epochs):
        ckpt_path = Path(self.config.RUN_DIR) / "ckpt"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        def run_epoch(mode: str, epoch):
            self.tqdm_vis = False
            is_train = False
            if mode == "train":
                is_train = True
                self.tqdm_vis = True
                
            intermediate_losses = defaultdict(float)
            
            self.tokenizer.train(is_train)
            self.model.train(is_train)
            
            tokenizer_aver_loss = 0.
            model_aver_loss = 0.
            
            # train tokenizer
            pbar = tqdm(range(self.config.MODEL_EPOCHS), desc=f"Epoch {epoch} - Training tokenizer", file=sys.stdout, disable=not self.tqdm_vis)
            for _ in pbar:
                samples = self.replay_buffer.sample_batch(batch_num_samples=256,
                                                          sequence_length=1,
                                                          sample_from_start=True)
                samples = self._to_device(samples)
                loss, loss_dict = self.tokenizer.compute_loss(samples["observation"])
                if is_train:
                    self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
                    
                tokenizer_aver_loss += loss.item()
                
                if self.config.tokenizer_type == 'vq':
                    pbar.set_description(
                        f"Epoch {epoch} - Training tokenizer:"
                        + f"rec loss: {loss_dict[self.config.tokenizer_type + '/rec_loss']:.4f} | "
                        + f"cmt loss: {loss_dict[self.config.tokenizer_type + '/cmt_loss']:.4f} | "
                        + f"active %: {loss_dict[self.config.tokenizer_type + '/active']:.3f} | "
                    )
                elif self.config.tokenizer_type == 'fsq':
                    pbar.set_description(
                        f"Epoch {epoch} - Training tokenizer:"
                        + f"rec loss: {loss_dict[self.config.tokenizer_type + '/rec_loss']:.4f} | "
                        + f"active %: {loss_dict[self.config.tokenizer_type + '/active']:.3f} | "
                    )
                else:
                    raise NotImplementedError
                    
                for loss_name, loss_value in loss_dict.items():
                    intermediate_losses[loss_name + "_" + mode] += loss_value / self.config.MODEL_EPOCHS
            
            # train world model
            self.tokenizer.eval()
            if (epoch + 1) >= 10:
                pbar = tqdm(range(self.config.MODEL_EPOCHS), desc=f"Epoch {epoch} - Training {str(self.model)}", file=sys.stdout, disable=not self.tqdm_vis)
                for _ in pbar:
                    samples = self.replay_buffer.sample_batch(batch_num_samples=self.config.MODEL_BATCH_SIZE,
                                                            sequence_length=self.config.SEQ_LENGTH,
                                                            sample_from_start=True)
                    samples = self._to_device(samples)
                    loss, loss_dict = self.model.compute_loss(samples, self.tokenizer)
                    if is_train:
                        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.max_grad_norm)
                    
                    model_aver_loss += loss.item()

                    for loss_name, loss_value in loss_dict.items():
                        intermediate_losses[loss_name + "_" + mode] += loss_value / self.config.MODEL_EPOCHS

                    pbar.set_description(
                        f"Epoch {epoch} - Training world model:"
                        + f"total loss: {loss_dict['world_model/total_loss']:.4f} | "
                        + f"obs loss: {loss_dict['world_model/loss_obs']:.4f} | "
                        + f"rew loss: {loss_dict['world_model/loss_rewards']:.4f} | "
                        + f"dis loss: {loss_dict['world_model/loss_ends']:.3f} | "
                        + f"av loss: {loss_dict['world_model/loss_av_actions']:.3f} | "
                    )
            
            
            wandb.log({'epoch': epoch, **intermediate_losses})
            
            return tokenizer_aver_loss / self.config.MODEL_EPOCHS, model_aver_loss / self.config.MODEL_EPOCHS
        
        
        for i in range(epochs):
            t_loss, wm_loss = run_epoch("train", i)
            
            if (i + 1) > 10:
                self.save(ckpt_path / f"epoch_{i}.pth")
            
            if (i + 1) % 20 == 0:
                self.model.eval()
                self.tokenizer.eval()
                
                sample = self.replay_buffer.sample_batch(batch_num_samples=1,
                                                         sequence_length=self.config.HORIZON,
                                                         sample_from_start=True,
                                                         valid_sample=True)
                sample = self._to_device(sample)
                self.model.visualize_attn(sample, self.tokenizer, Path(self.config.RUN_DIR) / "visualization" / "attn" / f"epoch_{i + 1}")
    
    
    ### for training actor and critic only
    def train_actor_only(self, rollout):
        #### adding data to replay buffer
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])

        self.add_experience_to_dataset(rollout)
        self.mamba_replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                        rollout['fake'], rollout['last'], rollout.get('avail_action'))
        
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if self.replay_buffer.num_steps < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        self.train_count += 1

        intermediate_losses = defaultdict(float)
        
        # train actor-critic
        for i in tqdm(range(self.config.EPOCHS), desc=f"Training actor-critic", file=sys.stdout, disable=not self.tqdm_vis):
            samples = self.replay_buffer.sample_batch(batch_num_samples=self.config.MODEL_BATCH_SIZE * 20, # self.config.MODEL_BATCH_SIZE * 2
                                                        sequence_length=self.config.stack_obs_num if self.config.use_stack else 1,
                                                        sample_from_start=False,
                                                        valid_sample=False)
            samples = self._to_device(samples)
            self.train_agent_with_transformer(samples)

        wandb.log({'epoch': self.cur_wandb_epoch, **intermediate_losses})