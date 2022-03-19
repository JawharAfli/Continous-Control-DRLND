REACHER_ENVIRONMENT = "/home/jawhar/Desktop/udacity/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux/Reacher.x86_64"
DEVICE = "cuda:0"

N_EPISODES = 2000
MAX_T = 1000
UPDATE_EVERY = 10


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
