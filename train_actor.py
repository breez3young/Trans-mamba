import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime

from agent.runners.DreamerRunner import DreamerRunner
from configs import Experiment, SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, RewardsComposerConfig
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig
from configs.flatland.RewardConfigs import FinishRewardConfig
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from configs.flatland.TimetableConfigs import AllAgentLauncherConfig
from env.flatland.params import SeveralAgents, PackOfAgents, LotsOfAgents
from environments import Env, FlatlandType, FLATLAND_OBS_SIZE, FLATLAND_ACTION_SIZE
from utils import generate_group_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="starcraft", help='Flatland or SMAC env')
    parser.add_argument('--map', type=str, default="3m", help='Specific setting')
    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1e6)
    parser.add_argument('--mode', type=str, default='disabled')
    parser.add_argument('--tokenizer', type=str, default='vq')
    parser.add_argument('--decay', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1.)
    
    parser.add_argument('--minibuffer_size', type=float, default=500)
    return parser.parse_args()


def train_actor_with_pretrained_wm(exp, n_workers, load_path): 
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.train_actor(load_path, exp.steps, exp.episodes)


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


if __name__ == "__main__":
    RANDOM_SEED = 23
    args = parse_args()
    RANDOM_SEED += args.seed * 100
    if args.env == Env.FLATLAND:
        configs = prepare_flatland_configs(args.map)
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.map)
    else:
        raise Exception("Unknown environment")
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    configs["learner_config"].tokenizer_type = args.tokenizer
    configs["controller_config"].tokenizer_type = args.tokenizer
    configs["learner_config"].ema_decay = args.decay
    configs["controller_config"].ema_decay = args.decay
    
    configs["controller_config"].temperature = args.temperature
    
    ## preparing for training actor only
    configs["learner_config"].load_path = None
    configs["learner_config"].is_preload = False
    configs["learner_config"].use_external_rew_model = False
    configs["learner_config"].MIN_BUFFER_SIZE = args.minibuffer_size

    # make run directory
    save_path = Path("pretrained_weights") / f'actor_with_pretrained_wm' / (f"mawm_{args.map}_actor_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3])
    save_path.mkdir(parents=True, exist_ok=True)
    save_ckpt_path = save_path / 'ckpt'
    save_ckpt_path.mkdir(parents=True, exist_ok=True)
    
    configs["learner_config"].RUN_DIR = str(save_path)
    
    print(f"Run files are saved at {str(save_path)}")
    # -------------------
    
    # setting the world model path to load
    args.wm_path = "/mnt/data/optimal/zhangyang/code/bins/pretrained_weights/pretrained_wm/mawm_2m_vs_1z_vq_50K_obs16_2024-03-17_00-04-45-325/ckpt/epoch_499.pth"
    

    global wandb
    import wandb
    wandb.init(
        config=vars(args),
        mode=args.mode,
        project="0301_sc2",
        group="(actor w/ pretrained wm)" + f"mawm_actor_temp={configs['controller_config'].temperature}",
        name=f'mawm_{args.map}_seed_{RANDOM_SEED}',
    )

    exp = Experiment(steps=args.steps,
                     episodes=50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_actor_with_pretrained_wm(exp, n_workers=args.n_workers, load_path=args.wm_path)
