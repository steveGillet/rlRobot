from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from robotArmEnv import robotArmEnv, setupLogging
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faulthandler

faulthandler.enable(file=open(f"logs/faulthandler{os.getpid()}.log", "w"))
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []  # Buffer recent rewards

    def _on_step(self) -> bool:
        # Collect reward from the latest step (across vec envs)
        if "rewards" in self.locals:
            self.rewards.extend(self.locals["rewards"])
        # Log mean every 100 steps (adjust as needed)
        if self.num_timesteps % 100 == 0 and self.rewards:
            mean_reward = sum(self.rewards) / len(self.rewards)
            self.logger.record("custom/mean_reward", mean_reward)
            self.rewards = []  # Reset buffer
        return True


def makeEnv(taskName):
    def _init():
        return robotArmEnv(taskName=taskName)

    return _init


if __name__ == "__main__":
    logger = setupLogging()
    logger.info("Main Process Started")
    activeTask = "sideToSide"

    venv = SubprocVecEnv([makeEnv(activeTask) for _ in range(24)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

    policyKwargs = dict(net_arch=[256, 256, 256])   

    model = SAC(         
        "MlpPolicy",
        venv,
        ent_coef="auto",                 
        verbose=1,
        policy_kwargs=policyKwargs,
        learning_rate=3e-4,       
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=f"./arm_morph_tb_{activeTask}_SAC/",
        device="cpu",
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=RewardLoggerCallback(),
        # log_interval=10
    )
    modelName = "SACbaseJointLimit"
    venv.save(f"{activeTask}_{modelName}_vecnormalize.pkl")
    model.save(f"{activeTask}_{modelName}_1_100_10_1_0.0001")

# if __name__ == "__main__":
#     env = robotArmEnv(taskName="container")
#     obs, info = env.reset()
#     for _ in range(2500):
#         a = env.action_space.sample()
#         print(a)
#         print("step...")
#         obs, r, done, trunc, info = env.step(a)
#         print("reward:", r)
#         obs, info = env.reset()

#     print("DONE WITH MAIN PYTHON CODE")
#     import os
#     os._exit(0)
