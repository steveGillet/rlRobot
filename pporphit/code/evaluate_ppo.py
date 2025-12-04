import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robotArmEnv import robotArmEnv, ik_dls, manipulabilityIndex, generateXML
import pandas as pd
import os
import time
import mujoco


def evaluate_policy(
    model_path, num_episodes=10, output_file="ppo_evaluation_metrics.csv"
):
    """
    Evaluates a trained PPO policy on the robotArmEnv.

    Args:
        model_path (str): Path to the trained PPO model (without .zip extension).
        num_episodes (int): Number of episodes to evaluate.
        output_file (str): Path to save the evaluation metrics CSV.
    """

    print(f"Loading model from {model_path}...")

    # Load the environment
    # We need to wrap it in DummyVecEnv and VecNormalize to match training
    # Note: For accurate evaluation, we should ideally load the normalization stats from training.
    # If a stats file exists (e.g., vec_normalize.pkl), we should load it.
    # For now, we'll create a fresh VecNormalize, but be aware that without the training stats,
    # the agent might perform slightly differently if observations were heavily normalized.
    # However, since observation is just [0.0], normalization doesn't impact input much,
    # but reward normalization does. We turn off training/updating stats.

    env = DummyVecEnv([lambda: robotArmEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, training=False)

    # Load the model
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    metrics = []

    print(f"Starting evaluation for {num_episodes} episodes...")

    for episode in range(num_episodes):
        start_time = time.time()
        obs = env.reset()
        done = False

        # In this specific env, one step = one episode (design -> evaluate)
        # But we'll write a loop just in case the env changes

        # Get action from policy
        # deterministic=True is usually better for evaluation
        action, _ = model.predict(obs, deterministic=False)

        # Step the environment
        # The environment returns the reward for the design
        obs, rewards, dones, infos = env.step(action)

        # Since it's a VecEnv, rewards/dones are arrays
        reward = rewards[0]

        # We need to decode the action to get the morphology details for logging
        # We can use the logic from robotArmEnv or visualize_ppo_morph
        # Re-implementing decode here for logging purposes
        raw_action = action[0]
        minNumLinks = 4
        maxNumLinks = 4
        minLength = 0.05
        maxLength = 1.2

        # Decode number of links
        numLinks = int(
            np.round(raw_action[0] * (maxNumLinks - minNumLinks) + minNumLinks)
        )

        # Decode lengths (just for the active links)
        link_action_segment = raw_action[1 : (maxNumLinks + 1)]
        decoded_lengths = (link_action_segment * (maxLength - minLength) + minLength)[
            :numLinks
        ]
        print(decoded_lengths)
        total_length = np.sum(decoded_lengths)

        # Decode joint types
        joint_type_action_segment = raw_action[(1 + maxNumLinks) :]
        decoded_joint_types = np.round(joint_type_action_segment * 3)[:numLinks].astype(
            int
        )

        duration = time.time() - start_time

        # Calculate manipulability
        # We need to create a MuJoCo model for this specific morphology
        try:
            startPos = np.array([0.41, 0.21, 0.3])
            goalPos = np.array([0.45, 0.25, 0.65])

            # Use the correct 3-parameter generateXML
            xml = generateXML(
                numLinks, decoded_lengths.tolist(), decoded_joint_types.tolist()
            )
            model_temp = mujoco.MjModel.from_xml_string(xml)
            data_temp = mujoco.MjData(model_temp)

            # Compute IK for start and goal
            startQpos, jStart = ik_dls(model_temp, startPos)
            goalQpos, jGoal = ik_dls(model_temp, goalPos, initialQpos=startQpos)

            # Calculate manipulability from Jacobian
            # Manipulability = sqrt(det(J * J^T))
            def calc_manipulability(jacobian):
                if jacobian is None or not np.all(np.isfinite(jacobian)):
                    return 0.0
                # Jacobian is 3xN (3 DOF in task space, N joints)
                # We only care about the position part (first 3 rows)
                J_pos = jacobian[:3, :]  # Position Jacobian (3 x numLinks)
                JJT = J_pos @ J_pos.T  # 3x3 matrix
                det = np.linalg.det(JJT)
                if det <= 0:
                    return 0.0
                return np.sqrt(det)

            if startQpos is not None and jStart is not None:
                muStart = calc_manipulability(jStart)
            else:
                muStart = 0.0

            if goalQpos is not None and jGoal is not None:
                muGoal = calc_manipulability(jGoal)
            else:
                muGoal = 0.0

        except Exception as e:
            print(f"  Warning: Could not calculate manipulability: {e}")
            muStart = 0.0
            muGoal = 0.0

        print(
            f"Episode {episode+1}/{num_episodes}: Reward={reward:.4f}, Links={numLinks}, Total Length={total_length:.4f}m, Manip(Start)={muStart:.4f}, Manip(Goal)={muGoal:.4f}"
        )

        metrics.append(
            {
                "Episode": episode + 1,
                "Reward": reward,
                "NumLinks": numLinks,
                "TotalLength": total_length,
                "JointTypes": str(decoded_joint_types.tolist()),
                "LinkLengths": str(np.round(decoded_lengths, 4).tolist()),
                "ManipulabilityStart": muStart,
                "ManipulabilityGoal": muGoal,
                "Duration": duration,
            }
        )

    # Save to CSV
    df = pd.DataFrame(metrics)
    df.to_csv(output_file, index=False)
    print(f"\nEvaluation complete. Metrics saved to {output_file}")

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Average Reward: {df['Reward'].mean():.4f} +/- {df['Reward'].std():.4f}")
    print(f"Average Num Links: {df['NumLinks'].mean():.2f}")
    print(f"Average Total Length: {df['TotalLength'].mean():.4f} m")


if __name__ == "__main__":
    # Assumes the model is named 'shelfArm' and is in the current directory
    MODEL_NAME = "shelfArm"
    evaluate_policy(MODEL_NAME, num_episodes=10)
