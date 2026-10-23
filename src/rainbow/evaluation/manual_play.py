import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
from config import config
from ..factory import Factory


factory = Factory(config)
env = factory.create_manual_play_environment()

BREAKOUT_KEYS_TO_ACTION = {
    (): 0,          # NOOP
    (" ",): 1,      # FIRE
    ("a",): 3,      # LEFT
    ("d",): 2,      # RIGHT
}

play(env, keys_to_action=BREAKOUT_KEYS_TO_ACTION)




