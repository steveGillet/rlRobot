import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from robotArmEnv import robotArmEnv
from simple_rl import PolicyNetwork, get_env_action, NUM_LINKS, MIN_LENGTH, MAX_LENGTH

import glob
import os


def get_panda_action():
    env = robotArmEnv()

    target_num_links = 7
    target_lengths = np.array([0.333, 0.0825, 0.316, 0.0825, 0.384, 0.088, 0.01])
    target_types = np.array([2, 1, 2, 1, 0, 1, 2])

    action = np.zeros(1 + env.maxNumLinks * 2, dtype=np.float32)
    action[0] = 1.0
    for i, l in enumerate(target_lengths):
        if i < env.maxNumLinks:
            action[1 + i] = (l - env.minLength) / (env.maxLength - env.minLength)

    for i, t in enumerate(target_types):
        if i < env.maxNumLinks:
            action[1 + env.maxNumLinks + i] = t / 3.0

    return action


def evaluate_agent(agent_type, env, ppo_model=None, reinforce_policy=None):
    rewards = []
    approaches = []
    metrics_sum = {
        "reward": 0.0,
        "link_penalty": 0.0,
        "path_penalty": 0.0,
        "accuracy_penalty": 0.0,
        "manipulability_bonus": 0.0,
        "energy_cost": 0.0,
    }

    obs, info = env.reset()

    if agent_type == "PPO":
        action, _ = ppo_model.predict(obs, deterministic=True)
    elif agent_type == "REINFORCE":
        state = torch.FloatTensor([0.0])
        mean, _ = reinforce_policy(state)
        action_normalized = torch.clamp(mean, 0.0, 1.0).detach().numpy()
        action = get_env_action(action_normalized)
    elif agent_type == "Panda":
        action = get_panda_action()

    _, reward, done, _, _ = env.step(action)

    metrics = env.last_info.copy()
    metrics["reward"] = reward

    return metrics


def plot_results(results, filename):
    agents = list(results.keys())
    metric_names = [
        "reward",
        "link_penalty",
        "path_penalty",
        "accuracy_penalty",
        "manipulability_bonus",
    ]

    n_agents = len(agents)
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 5))

    colors = ["skyblue", "orange", "green", "red", "purple", "brown"]

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        values = [results[agent].get(metric, 0.0) for agent in agents]

        bars = ax.bar(agents, values, color=colors[:n_agents])

        ax.set_title(metric)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)

        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                round(yval, 2),
                va="bottom" if yval > 0 else "top",
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def plot_detailed_analysis(results, task_name):
    agents = list(results.keys())

    accuracies = [results[a].get("accuracy_penalty", -100) for a in agents]
    link_penalties = [results[a].get("link_penalty", -100) for a in agents]

    plt.figure(figsize=(10, 6))
    for i, agent in enumerate(agents):
        plt.scatter(link_penalties[i], accuracies[i], s=200, label=agent)

    plt.title(f"Complexity vs Capability Trade-off ({task_name})")
    plt.xlabel("Link Penalty (Hardware Complexity)")
    plt.ylabel("Accuracy Penalty (Task Performance)")
    plt.legend()
    plt.grid(True)

    filename = f"analysis_tradeoff_{task_name}.png"
    plt.savefig(filename)
    print(f"Trade-off plot saved to {filename}")
    plt.close()  # Close to free memory

    man_scores = [results[a].get("manipulability_bonus", 0) for a in agents]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        agents, man_scores, color=["skyblue", "orange", "green", "red"][: len(agents)]
    )
    plt.title(f"Robot Dextrous Potential (Manipulability) - {task_name}")
    plt.ylabel("Manipulability Index (Bonus)")
    plt.tick_params(axis="x", rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            round(yval, 2),
            va="bottom",
            ha="center",
        )

    filename = f"analysis_manipulability_{task_name}.png"
    plt.savefig(filename)
    print(f"Manipulability plot saved to {filename}")
    plt.close()


def main():
    env = robotArmEnv()

    tasks = {
        "Container_Task": ["containerArm", "bestestArm"],
        "Shelf_Task": ["shelfArm", "shelfPreManArm"],
        "TwoShelf_Task": ["twoShelfArm", "twoShelfArmDoubleRew"],
    }

    for task_name, model_names in tasks.items():
        print(f"Evaluating {task_name}")
        task_results = {}

        for model_name in model_names:
            zip_file = f"{model_name}.zip"
            if not os.path.exists(zip_file):
                print(f"Warning: {zip_file} not found, skipping.")
                continue

            print(f"Evaluating {model_name}...")
            try:
                model = PPO.load(zip_file)
                task_results[model_name] = evaluate_agent("PPO", env, ppo_model=model)
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
                
        print("Evaluating REINFORCE Baseline...")
        try:
            policy = PolicyNetwork(1, NUM_LINKS)
            policy.load_state_dict(torch.load("reinforce_policy.pth"))
            task_results["REINFORCE"] = evaluate_agent(
                "REINFORCE", env, reinforce_policy=policy
            )
        except Exception as e:
            print(f"Failed to evaluate REINFORCE: {e}")

        print("Evaluating Panda Baseline...")
        try:
            task_results["Panda"] = evaluate_agent("Panda", env)
        except Exception as e:
            print(f"Failed to evaluate Panda: {e}")

        if task_results:
            plot_results(task_results, f"ablation_{task_name}.png")
            plot_detailed_analysis(task_results, task_name)

            print(f"Results for {task_name}")
            for agent, metrics in task_results.items():
                print(
                    f"Agent: {agent}, Reward: {metrics['reward']:.2f}, Success: {metrics.get('success', 0.0):.2f}"
                )


if __name__ == "__main__":
    main()
