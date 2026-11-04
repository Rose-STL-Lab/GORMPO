from .gym import *
from .adroit import *


loaded_args = {
    # d4rl gym
    "halfcheetah-random-v2": halfcheetah_random_args,
    "hopper-random-v2": hopper_random_args,
    "walker2d-random-v2": walker2d_random_args,
    "halfcheetah-medium-v2": halfcheetah_medium_args,
    "hopper-medium-v2": hopper_medium_args,
    "walker2d-medium-v2": walker2d_medium_args,
    "halfcheetah-medium-replay-v2": halfcheetah_medium_replay_args,
    "hopper-medium-replay-v2": hopper_medium_replay_args,
    "walker2d-medium-replay-v2": walker2d_medium_replay_args,
    "halfcheetah-medium-expert-v2": halfcheetah_medium_expert_args,
    "hopper-medium-expert-v2": hopper_medium_expert_args,
    "walker2d-medium-expert-v2": walker2d_medium_expert_args,

    # d4rl adroit
    "pen-cloned-v1": pen_cloned_args,
    "pen-human-v1": pen_human_args,
}