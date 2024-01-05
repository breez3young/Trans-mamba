from agent.learners.DreamerLearner import DreamerLearner
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        # self.MODEL_LR = 2e-4
        # self.ACTOR_LR = 5e-4
        # self.VALUE_LR = 5e-4
        # self.CAPACITY = 500000 # 这个buffer的长度刚好就是想要训练到的最大步长
        # self.MIN_BUFFER_SIZE = 100
        # self.MODEL_EPOCHS = 1
        # self.EPOCHS = 1
        # self.PPO_EPOCHS = 5
        # self.MODEL_BATCH_SIZE = 40
        # self.BATCH_SIZE = 40
        # self.SEQ_LENGTH = 50
        # self.N_SAMPLES = 1
        # self.TARGET_UPDATE = 1
        # self.DEVICE = 'cpu'
        # self.GRAD_CLIP = 100.0
        # self.HORIZON = 15
        # self.ENTROPY = 0.001
        # self.ENTROPY_ANNEALING = 0.99998
        # self.GRAD_CLIP_POLICY = 100.

        # optimal smac config
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4
        self.VALUE_LR = 5e-4
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 500 # 500
        self.MODEL_EPOCHS = 200 # 60
        self.EPOCHS = 8 # 4; 27m epochs should be 20, agents_num ~ 10 should be 20
        self.PPO_EPOCHS = 5
        self.MODEL_BATCH_SIZE = 30 # 40; 27m bs should be 10, agents_num ~ 10 should be 20
        self.BATCH_SIZE = 30 # 40; 27m bs should be 8, agents_num ~ 10 should be 20
        # self.SEQ_LENGTH = 20
        self.SEQ_LENGTH = self.HORIZON
        self.N_SAMPLES = 1
        self.TARGET_UPDATE = 1
        self.DEVICE = 'cuda'
        self.GRAD_CLIP = 100.0
        # self.HORIZON = 15
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.999996 # 0.99998
        self.GRAD_CLIP_POLICY = 100.0

        # tokenizer
        ## batch size
        self.t_bs = 1024
        ## learning rate
        self.t_lr = 1e-4

        # world model
        ## batch size
        self.wm_bs = 64
        ## learning rate
        self.wm_lr = 1e-4 # 5e-4
        self.wm_weight_decay = 0.01

        self.max_grad_norm = 10.0

    def create_learner(self):
        return DreamerLearner(self)
