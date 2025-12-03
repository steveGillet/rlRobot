import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from stable_baselines3 import PPO
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


def makeEnv():
    def _init():
        return robotArmEnv()

    return _init


if __name__ == "__main__":
    logger = setupLogging()
    logger.info("Main Process Started")

    venv = SubprocVecEnv([makeEnv() for _ in range(24)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

    policyKwargs = dict(net_arch=[128, 128, 128])
    ppo = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policyKwargs,
        learning_rate=0.001,
        n_steps=128,
        batch_size=512,
        n_epochs=4,
        gamma=0.98,
        verbose=1,
        tensorboard_log="./arm_morph_tb/",
        device="cpu",
    )
    ppo.learn(total_timesteps=1_000_000, callback=RewardLoggerCallback())
    ppo.save("twoShelfArm")

# if __name__ == "__main__":
#     env = robotArmEnv()
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


# index = 0

# model.site('startPos').pos = startPos
# model.site('goalPos').pos = goalPos

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while index < len(pathStates):
#         i = 0
#         for id in jointIds:
#             data.qpos[id] = pathStates[index][i].value
#             i+=1

#         print(data.qpos[2])

#         mujoco.mj_step(model,data)
#         print(f"Time step: {data.time}s, Position: {data.geom_xpos[-1]}")
#         viewer.sync()
#         time.sleep(0.1)
#         index += 1

# print("Sim Complete")
