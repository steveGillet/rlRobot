from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from robotArmEnv import robotArmEnv
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def makeEnv():
    def _init():
        return robotArmEnv(4)
    return _init

if __name__ == '__main__':
    venv = SubprocVecEnv([makeEnv() for _ in range(16)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

    policyKwargs = dict(net_arch=[128,128,128])
    ppo = PPO("MlpPolicy", venv, policy_kwargs=policyKwargs, learning_rate=0.001, n_steps=128, batch_size=512, n_epochs=4, gamma=0.98, verbose=1, tensorboard_log="./arm_morph_tb/", device="cpu")
    ppo.learn(total_timesteps=3_000_000)
    ppo.save("bestestArm")

# if __name__ == "__main__":
#     env = robotArmEnv(4)
#     obs, info = env.reset()
#     for _ in range(25):
#         a = env.action_space.sample()
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