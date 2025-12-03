import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from tensorboard.backend.event_processing import event_accumulator
import os
import glob

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 12


def load_tensorboard_data(log_dir):
    """
    Load training data from TensorBoard logs.
    """
    # Find the most recent event file
    event_files = glob.glob(
        os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True
    )

    if not event_files:
        print(f"No TensorBoard logs found in {log_dir}")
        return None

    # Use the most recent event file
    event_file = max(event_files, key=os.path.getctime)
    print(f"Loading TensorBoard data from: {event_file}")

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Get available scalar tags
    tags = ea.Tags()["scalars"]
    print(f"Available metrics: {tags}")

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {"steps": steps, "values": values}

    return data


def plot_training_curves(tb_data, output_dir="plots"):
    """
    Plot training curves from TensorBoard data.
    """
    os.makedirs(output_dir, exist_ok=True)

    if tb_data is None:
        print("No TensorBoard data available for plotting.")
        return

    # Plot 1: Reward over time
    if "rollout/ep_rew_mean" in tb_data:
        plt.figure(figsize=(10, 6))
        steps = tb_data["rollout/ep_rew_mean"]["steps"]
        values = tb_data["rollout/ep_rew_mean"]["values"]
        plt.plot(steps, values, linewidth=2, color="#2E86AB")
        plt.xlabel("Training Steps")
        plt.ylabel("Mean Episode Reward")
        plt.title("PPO Training Progress: Mean Episode Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_reward.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir}/training_reward.png")

    # Plot 2: Episode length
    if "rollout/ep_len_mean" in tb_data:
        plt.figure(figsize=(10, 6))
        steps = tb_data["rollout/ep_len_mean"]["steps"]
        values = tb_data["rollout/ep_len_mean"]["values"]
        plt.plot(steps, values, linewidth=2, color="#A23B72")
        plt.xlabel("Training Steps")
        plt.ylabel("Mean Episode Length")
        plt.title("PPO Training Progress: Mean Episode Length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/training_episode_length.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved: {output_dir}/training_episode_length.png")

    # Plot 3: Learning rate
    if "train/learning_rate" in tb_data:
        plt.figure(figsize=(10, 6))
        steps = tb_data["train/learning_rate"]["steps"]
        values = tb_data["train/learning_rate"]["values"]
        plt.plot(steps, values, linewidth=2, color="#F18F01")
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("PPO Training Progress: Learning Rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/training_learning_rate.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved: {output_dir}/training_learning_rate.png")

    # Plot 4: Loss values (if available)
    loss_metrics = ["train/policy_loss", "train/value_loss", "train/entropy_loss"]
    available_losses = [m for m in loss_metrics if m in tb_data]

    if available_losses:
        plt.figure(figsize=(10, 6))
        for metric in available_losses:
            steps = tb_data[metric]["steps"]
            values = tb_data[metric]["values"]
            label = metric.split("/")[-1].replace("_", " ").title()
            plt.plot(steps, values, linewidth=2, label=label, alpha=0.8)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("PPO Training Progress: Loss Components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_losses.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir}/training_losses.png")


def plot_evaluation_metrics(csv_path, output_dir="plots"):
    """
    Plot evaluation metrics from the CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Evaluation CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\nEvaluation Metrics Summary:")
    print(df.describe())

    # Plot 1: Reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["Reward"], bins=20, color="#06A77D", alpha=0.7, edgecolor="black")
    plt.axvline(
        df["Reward"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["Reward"].mean():.2f}',
    )
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Distribution of Evaluation Rewards")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/eval_reward_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir}/eval_reward_distribution.png")

    # Plot 2: Reward over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["Episode"],
        df["Reward"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#2E86AB",
    )
    plt.axhline(
        df["Reward"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["Reward"].mean():.2f}',
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation Reward per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/eval_reward_per_episode.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir}/eval_reward_per_episode.png")

    # Plot 3: Number of links distribution
    plt.figure(figsize=(10, 6))
    link_counts = df["NumLinks"].value_counts().sort_index()
    plt.bar(
        link_counts.index,
        link_counts.values,
        color="#A23B72",
        alpha=0.7,
        edgecolor="black",
    )
    plt.xlabel("Number of Links")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Links in Evaluated Designs")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/eval_num_links_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir}/eval_num_links_distribution.png")

    # Plot 4: Total length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["TotalLength"], bins=20, color="#F18F01", alpha=0.7, edgecolor="black")
    plt.axvline(
        df["TotalLength"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["TotalLength"].mean():.3f}m',
    )
    plt.xlabel("Total Length (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Robot Length")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/eval_total_length_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir}/eval_total_length_distribution.png")

    # Plot 5: Reward vs Total Length scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["TotalLength"],
        df["Reward"],
        alpha=0.6,
        s=100,
        color="#06A77D",
        edgecolor="black",
    )
    plt.xlabel("Total Length (m)")
    plt.ylabel("Reward")
    plt.title("Reward vs Total Robot Length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eval_reward_vs_length.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/eval_reward_vs_length.png")

    # Plot 6: Manipulability at Start and Goal
    if "ManipulabilityStart" in df.columns and "ManipulabilityGoal" in df.columns:
        plt.figure(figsize=(12, 5))

        # Subplot 1: Manipulability comparison
        plt.subplot(1, 2, 1)
        x = np.arange(len(df))
        width = 0.35
        plt.bar(
            x - width / 2,
            df["ManipulabilityStart"],
            width,
            label="Start",
            color="#2E86AB",
            alpha=0.7,
            edgecolor="black",
        )
        plt.bar(
            x + width / 2,
            df["ManipulabilityGoal"],
            width,
            label="Goal",
            color="#A23B72",
            alpha=0.7,
            edgecolor="black",
        )
        plt.xlabel("Episode")
        plt.ylabel("Manipulability Index")
        plt.title("Manipulability at Start vs Goal Positions")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        # Subplot 2: Manipulability distribution
        plt.subplot(1, 2, 2)
        plt.hist(
            df["ManipulabilityStart"],
            bins=15,
            alpha=0.6,
            label="Start",
            color="#2E86AB",
            edgecolor="black",
        )
        plt.hist(
            df["ManipulabilityGoal"],
            bins=15,
            alpha=0.6,
            label="Goal",
            color="#A23B72",
            edgecolor="black",
        )
        plt.xlabel("Manipulability Index")
        plt.ylabel("Frequency")
        plt.title("Distribution of Manipulability")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/eval_manipulability.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved: {output_dir}/eval_manipulability.png")

        # Plot 7: Manipulability vs Total Length
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(
            df["TotalLength"],
            df["ManipulabilityStart"],
            alpha=0.6,
            s=100,
            color="#2E86AB",
            edgecolor="black",
        )
        plt.xlabel("Total Length (m)")
        plt.ylabel("Manipulability Index")
        plt.title("Manipulability (Start) vs Total Length")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(
            df["TotalLength"],
            df["ManipulabilityGoal"],
            alpha=0.6,
            s=100,
            color="#A23B72",
            edgecolor="black",
        )
        plt.xlabel("Total Length (m)")
        plt.ylabel("Manipulability Index")
        plt.title("Manipulability (Goal) vs Total Length")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/eval_manipulability_vs_length.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved: {output_dir}/eval_manipulability_vs_length.png")


def generate_summary_report(model_path, csv_path, tb_log_dir, output_dir="plots"):
    """
    Generate a comprehensive summary report.
    """
    os.makedirs(output_dir, exist_ok=True)

    report = []
    report.append("=" * 80)
    report.append("PPO ROBOT MORPHOLOGY OPTIMIZATION - SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")

    # Model information
    if os.path.exists(f"{model_path}.zip"):
        report.append("MODEL INFORMATION:")
        report.append(f"  Model Path: {model_path}.zip")
        try:
            model = PPO.load(model_path)
            report.append(f"  Policy Architecture: {model.policy}")
            report.append(f"  Learning Rate: {model.learning_rate}")
            report.append(f"  Gamma: {model.gamma}")
        except Exception as e:
            report.append(f"  Error loading model: {e}")
        report.append("")

    # Evaluation metrics
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        report.append("EVALUATION METRICS:")
        report.append(f"  Number of Episodes: {len(df)}")
        report.append(
            f"  Mean Reward: {df['Reward'].mean():.4f} ± {df['Reward'].std():.4f}"
        )
        report.append(f"  Max Reward: {df['Reward'].max():.4f}")
        report.append(f"  Min Reward: {df['Reward'].min():.4f}")
        report.append(f"  Mean Number of Links: {df['NumLinks'].mean():.2f}")
        report.append(
            f"  Mean Total Length: {df['TotalLength'].mean():.4f}m ± {df['TotalLength'].std():.4f}m"
        )

        if "ManipulabilityStart" in df.columns and "ManipulabilityGoal" in df.columns:
            report.append(
                f"  Mean Manipulability (Start): {df['ManipulabilityStart'].mean():.4f} ± {df['ManipulabilityStart'].std():.4f}"
            )
            report.append(
                f"  Mean Manipulability (Goal): {df['ManipulabilityGoal'].mean():.4f} ± {df['ManipulabilityGoal'].std():.4f}"
            )

        report.append("")

    # Save report
    report_text = "\n".join(report)
    print("\n" + report_text)

    with open(f"{output_dir}/summary_report.txt", "w") as f:
        f.write(report_text)
    print(f"\nSaved: {output_dir}/summary_report.txt")


def main():
    # Configuration
    MODEL_PATH = "shelfArm"
    EVAL_CSV = "ppo_evaluation_metrics.csv"
    TB_LOG_DIR = "arm_morph_tb"
    OUTPUT_DIR = "plots"

    print("=" * 80)
    print("PPO ANALYSIS AND VISUALIZATION SCRIPT")
    print("=" * 80)
    print()

    # Load and plot TensorBoard data
    print("Loading TensorBoard training data...")
    tb_data = load_tensorboard_data(TB_LOG_DIR)
    plot_training_curves(tb_data, OUTPUT_DIR)
    print()

    # Plot evaluation metrics
    print("Plotting evaluation metrics...")
    plot_evaluation_metrics(EVAL_CSV, OUTPUT_DIR)
    print()

    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(MODEL_PATH, EVAL_CSV, TB_LOG_DIR, OUTPUT_DIR)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"All plots and reports saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
