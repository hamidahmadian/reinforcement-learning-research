import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epsilon variables
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (
        epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# misc agent variables
GAMMA = 0.99
LR = 1e-4

# memory
TARGET_NET_UPDATE_FREQ = 1000
EXP_REPLAY_SIZE = 100000
BATCH_SIZE = 32

# Learning control variables
LEARN_START = 10000
MAX_FRAMES = 1000000
UPDATE_FREQ = 1

# data logging parameters
ACTION_SELECTION_COUNT_FREQUENCY = 1000
