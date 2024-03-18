import ray
import wandb
from copy import deepcopy

from agent.workers.DreamerWorker import DreamerWorker
import ipdb

class DreamerServer:
    def __init__(self, n_workers, env_config, controller_config, model):
        ray.init()

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]
        
        eval_controller_config = deepcopy(controller_config)
        eval_controller_config.temperature = 1.0
        self.eval_episodes_num = 10
        self.eval_workers = [DreamerWorker.remote(i, env_config, eval_controller_config) for i in range(self.eval_episodes_num)]
        self.eval_tasks = []

    def append(self, idx, update):
        self.tasks.append(self.workers[idx].run.remote(update))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs
    
    ## eval
    def eval_append(self, idx, update):
        self.eval_tasks.append(self.eval_workers[idx].run.remote(update))
        
    def evaluate(self, model_params):
        eval_win_rate = 0.
        eval_returns = 0.
        
        for i in range(self.eval_episodes_num):
            self.eval_append(i, model_params)

        for i in range(self.eval_episodes_num):
            # self.eval_append(i, model_params)
            done_id, eval_tasks = ray.wait(self.eval_tasks)
            
            self.eval_tasks = eval_tasks
            eval_rollout, eval_info = ray.get(done_id)[0]
            
            eval_win_rate += eval_info["reward"]
            eval_returns += eval_rollout["reward"].sum(0).mean()
        
        return eval_win_rate / self.eval_episodes_num, eval_returns / self.eval_episodes_num


class DreamerRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0
        save_interval_steps = 0
        last_save_steps = 0
        last_eval_steps = 0

        wandb.define_metric("steps")
        wandb.define_metric("reward", step_metric="steps")

        while True:
            rollout, info = self.server.run()
            ent = rollout['entropy'].sum(0) / (rollout['entropy'] > 1e-6).sum(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            save_interval_steps += info["steps_done"]
            returns = rollout["reward"].sum(0).mean()

            wandb.log({'reward': info["reward"], 'steps': cur_steps})
            wandb.log({'returns': returns, "episodes": cur_episode})

            print("%4s" % cur_episode, "%5s" % (cur_steps), info["reward"], 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')

            self.learner.step(rollout)

            if (save_interval_steps - last_save_steps) > 1000:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{save_interval_steps // 1000}Ksteps.pth")
                last_save_steps = save_interval_steps // 1000 * 1000

            ## evaluation
            if (save_interval_steps - last_eval_steps) > 200:
                eval_win_rate, eval_returns = self.server.evaluate(self.learner.params())
                last_eval_steps = save_interval_steps // 200 * 200
                
                wandb.log({'eval_win_rate': eval_win_rate, "steps": save_interval_steps})
                wandb.log({'eval_returns': eval_returns, "steps": save_interval_steps})

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            
            self.server.append(info['idx'], self.learner.params())
    
    
    # only train the actor and critic
    def train_actor(self, world_model_path, max_steps=10 ** 10, max_episodes=10 ** 10):
        ## preload world model
        self.learner.load_pretrained_wm(world_model_path)
        
        cur_steps, cur_episode = 0, 0
        save_interval_steps = 0
        last_save_steps = 0

        wandb.define_metric("steps")
        wandb.define_metric("win rate", step_metric="steps")
        
        while True:
            rollout, info = self.server.run()
            ent = rollout['entropy'].sum(0) / (rollout['entropy'] > 1e-6).sum(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            save_interval_steps += info["steps_done"]
            returns = rollout["reward"].sum(0).mean()

            wandb.log({'win rate': info["reward"], 'steps': cur_steps})
            wandb.log({'returns': returns, "episodes": cur_episode})

            print("%4s" % cur_episode, "%5s" % (cur_steps), info["reward"], 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')

            # train actor only
            self.learner.train_actor_only(rollout)

            if (save_interval_steps - last_save_steps) > 1000:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{save_interval_steps // 1000}Ksteps.pth")
                last_save_steps = save_interval_steps // 1000 * 1000

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            
            self.server.append(info['idx'], self.learner.params())
        
        