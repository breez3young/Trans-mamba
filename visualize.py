import argparse
import os
import shutil
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from pathlib import Path
from utils import load_mamba_model, load_mawm_model, _wrap
from smac.env import StarCraft2Env
from copy import deepcopy
import numpy as np
import logging

from configs.flatland.TimetableConfigs import AllAgentLauncherConfig
from env.flatland.params import SeveralAgents, PackOfAgents, LotsOfAgents
from environments import Env, FlatlandType, FLATLAND_OBS_SIZE, FLATLAND_ACTION_SIZE
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig

from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="starcraft", help='Flatland or SMAC env')
    parser.add_argument('--map_name', type=str, default="2m_vs_1z", help='Specific setting')
    parser.add_argument('--tokenizer', type=str, default='vq')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--eval_episodes', type=int, default=1)
    return parser.parse_args()


# prepare environment
def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
    
    print(f'Observation dims: {env.n_obs}')
    print(f'Action dims: {env.n_actions}')
    print(f'Num agents: {env.n_agents}')
    env.close()


def get_env_info_flatland(configs):
    for config in configs:
        config.IN_DIM = FLATLAND_OBS_SIZE
        config.ACTION_SIZE = FLATLAND_ACTION_SIZE


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_flatland_configs(env_name):
    if env_name == FlatlandType.FIVE_AGENTS:
        env_config = SeveralAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.TEN_AGENTS:
        env_config = PackOfAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.FIFTEEN_AGENTS:
        env_config = LotsOfAgents(RANDOM_SEED + 100)
    else:
        raise Exception("Unknown flatland environment")
    obs_builder_config = SimpleObservationConfig(max_depth=3, neighbours_depth=3,
                                                 timetable_config=AllAgentLauncherConfig())
    reward_config = RewardsComposerConfig((FinishRewardConfig(finish_value=10),
                                           NearRewardConfig(coeff=0.01),
                                           DeadlockPunishmentConfig(value=-5)))
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    get_env_info_flatland(agent_configs)
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": reward_config,
            "obs_builder_config": obs_builder_config}

# ----------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    RANDOM_SEED = 12345
    if args.env == Env.FLATLAND:
        raise NotImplementedError("Currently, visulization does not support FLATLAND env.")
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.map_name)
    else:
        raise Exception("Unknown environment")
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    configs["learner_config"].tokenizer_type = args.tokenizer
    configs["controller_config"].tokenizer_type = args.tokenizer
    
    device = configs["learner_config"].DEVICE

    # loading model
    if "mamba" in args.model_path:
        replay_prefix = "mamba"
        model = load_mamba_model(configs["learner_config"], args.model_path)
        
        @torch.no_grad()
        def select_actions(obser, avail_action, prev_actions, prev_rnn_state):
            obser = obser.unsqueeze(0)
            avail_action = avail_action.unsqueeze(0)
            
            state = model["model"](obser, prev_actions, prev_rnn_state, None)
            feats = state.get_features()
            
            action, pi = actor(feats)
            pi[avail_action == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()
            
            return action, deepcopy(state)
        
    else:
        replay_prefix = "mawm"
        model = load_mawm_model(configs["learner_config"], args.model_path)
        
        @torch.no_grad()
        def select_actions(obser, avail_action):
            if not configs["learner_config"].use_bin:
                feats = model["tokenizer"].encode_decode(obser, True, True)
            else:
                tokens = discretize_into_bins(obser, configs["learner_config"].bins)
                feats = bins2continuous(tokens, configs["learner_config"].bins)
            
            action, pi = actor(feats)
            pi[avail_action == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()
            
            return action
            
    
    actor = model["actor"]
    
    # initialize raw env
    env = StarCraft2Env(
        map_name=args.map_name,
        continuing_episode=True,
        seed=RANDOM_SEED,
        replay_prefix=replay_prefix + f"_{args.map_name}",
        replay_dir="/mnt/data/optimal/zhangyang/SC2_Replay",
    )

    # reset env
    for idx in range(args.eval_episodes):
        rewards = []
        prev_rnn_state = None
        prev_actions = None
        
        obs, _ = env.reset()
        obs = torch.tensor(np.array(obs)).to(device)
        done = False
        
        # sample a trajectory
        while not done:
            av_action = torch.tensor(env.get_avail_actions()).to(device)

            if replay_prefix == "mamba":
                action, prev_rnn_state = select_actions(obs, av_action, prev_actions, prev_rnn_state)
                prev_actions = action.clone()
                action = action.squeeze(0)
            else:
                action = select_actions(obs, av_action)
            
            reward, done, info = env.step([ac.argmax() for i, ac in enumerate(action)])
            
            obs = torch.tensor(np.array(env.get_obs())).to(device)
            
            rewards.append(reward)
            
        print(
            f"Visualize {idx}th episode - "
            + f"take {len(rewards)} timesteps | "
            + f"meet episode limit = {info.get('episode_limit', False)} | "
            + f"returns: {np.sum(rewards)}"
        )
    
    env.save_replay()
    env.close()
