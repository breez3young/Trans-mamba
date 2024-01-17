from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

from agent.models.tokenizer import StateEncoderConfig
from agent.models.transformer import PerceiverConfig, TransformerConfig

from functools import partial

RSSM_STATE_MODE = 'discrete'


class DreamerConfig(Config):
    def __init__(self):
        super().__init__()
        # self.HIDDEN = 256
        # self.MODEL_HIDDEN = 256
        # self.EMBED = 256
        # self.N_CATEGORICALS = 32
        # self.N_CLASSES = 32
        # self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        # self.DETERMINISTIC = 256
        # self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        # self.GLOBAL_FEAT = self.FEAT + self.EMBED
        # self.VALUE_LAYERS = 2
        # self.VALUE_HIDDEN = 256
        # self.PCONT_LAYERS = 2
        # self.PCONT_HIDDEN = 256
        # self.ACTION_SIZE = 9
        # self.ACTION_LAYERS = 2
        # self.ACTION_HIDDEN = 256
        # self.REWARD_LAYERS = 2
        # self.REWARD_HIDDEN = 256
        # self.GAMMA = 0.99
        # self.DISCOUNT = 0.99
        # self.DISCOUNT_LAMBDA = 0.95
        # self.IN_DIM = 30
        self.LOG_FOLDER = 'wandb/'

        # optimal smac config
        self.HIDDEN = 256
        self.MODEL_HIDDEN = 256
        self.EMBED = 256
        self.N_CATEGORICALS = 32
        self.N_CLASSES = 32
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        self.DETERMINISTIC = 256
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 256
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 256
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 2
        self.ACTION_HIDDEN = 256
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 256
        self.GAMMA = 0.99
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95
        self.IN_DIM = 30

        # tokenizer params
        self.nums_obs_token = 12 # 4
        self.hidden_sizes = [512, 512]
        self.alpha = 1.0
        self.EMBED_DIM = 128 # 128
        self.OBS_VOCAB_SIZE = 1024 # 512

        self.encoder_config_fn = partial(StateEncoderConfig,
            nums_obs_token=self.nums_obs_token, 
            hidden_sizes=self.hidden_sizes,
            alpha=1.0,
            z_channels=self.EMBED_DIM * self.nums_obs_token
        )
        
        # world model params
        self.HORIZON = 20
        self.TRANS_EMBED_DIM = 256 # 256
        self.HEADS = 4

        #### deprecated (original perceiver params)
        self.perattn_HEADS = 4
        self.DROPOUT = 0.1

        # self.perattn_config = PerAttnConfig(
        #     query_dim=self.TRANS_EMBED_DIM,
        #     context_dim=self.TRANS_EMBED_DIM,
        #     heads=self.perattn_HEADS, # self.HEADS
        #     dim_head=64,
        #     dropout=self.DROPOUT,
        # )
        #### -------------------------------------

        self.perceiver_config_fn = partial(PerceiverConfig,
            dim=self.TRANS_EMBED_DIM,
            latent_dim=self.TRANS_EMBED_DIM,
            depth=2,
            cross_heads=1,
            cross_dim_head=64,
            latent_heads=8,
            latent_dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.
        )

        self.trans_config = TransformerConfig(
            tokens_per_block=self.nums_obs_token + 1 + 1,
            max_blocks=self.HORIZON,
            attention='causal',
            num_layers=6, # 10
            num_heads=self.HEADS,
            embed_dim=self.TRANS_EMBED_DIM,
            embed_pdrop=self.DROPOUT,
            resid_pdrop=self.DROPOUT,
            attn_pdrop=self.DROPOUT,
        )

        # 这里修改一下
        # self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        self.FEAT = self.EMBED_DIM * self.nums_obs_token
        # self.critic_FEAT = self.TRANS_EMBED_DIM * self.nums_obs_token # self.TRANS_EMBED_DIM
        self.GLOBAL_FEAT = self.FEAT + self.EMBED

    def update(self):
        self.encoder_config = self.encoder_config_fn(state_dim=self.IN_DIM)
        self.perceiver_config = self.perceiver_config_fn(num_latents=self.NUM_AGENTS)


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]
