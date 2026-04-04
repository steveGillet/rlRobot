import numpy as np
from stable_baselines3 import SAC
from robotArmEnv import robotArmEnv   # your env class

if __name__ == "__main__":
    activeTask = "outreach"                     # change if you used a different task
    model_path = f"outreach_SACbaseJointLimit_1_100_10_1_0.0001"

    print("Loading SAC model (no VecNormalize needed)...")
    model = SAC.load(model_path)                 # no env= argument

    # Your env always gives obs = [0.0]
    obs = np.array([[0.0]], dtype=np.float32)    # shape (1, 1) because SB3 expects batch

    # Get the deterministic (best) action
    action, _ = model.predict(obs, deterministic=True)
    raw_action = action[0]                       # remove batch dim → shape (15,)

    print(f"\nRaw best action vector from SAC:\n{np.round(raw_action, 5)}\n")

    # ====================== DECODE MORPHOLOGY ======================
    env = robotArmEnv(taskName=activeTask)       # just for the constants

    numLinks = int(np.round(raw_action[0] * (env.maxNumLinks - env.minNumLinks) 
                            + env.minNumLinks))
    lengths = (raw_action[1 : (env.maxNumLinks + 1)] 
               * (env.maxLength - env.minLength) 
               + env.minLength)[:numLinks]
    jointTypes = np.round(raw_action[(1 + env.maxNumLinks):] * 3)[:numLinks].astype(int)

    print("=== BEST ROBOT MORPHOLOGY FOUND BY SAC ===")
    print(f"Number of links : {numLinks}")
    print(f"Link lengths    : {np.round(lengths, 4)}")
    print(f"Joint types     : {jointTypes}   (0=Rx, 1=Ry, 2=Rz, 3=Prismatic)")
    print("=" * 55)

    # ====================== FINAL EVALUATION ======================
    final_reward = env._evaluate(numLinks, lengths, jointTypes)
    print(f"✅ Final evaluated reward = {final_reward:.2f}")