# main_part3.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time  
import os    

from main_env_part3 import MazeEnv
from sarsa_lambda_agent import SarsaLambdaAgent

def clear_console():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def find_successful_episode(agent, env, render_on=False):
    """
    Runs episodes until one successfully reaches the goal.
    Returns the eligibility trace table from the final step of that episode.
    """
    print(f"Searching for a successful episode with lambda = {agent.lambda_val}...")
    
    env.reset()
    agent.reset_traces()
    terminated = False
    action = agent.choose_action(env._pos_to_state(env.agent_pos))
    state, reward, terminated, _, _ = env.step(action)

    while not terminated:
        nextAction = agent.choose_action(env._pos_to_state(env.agent_pos))
        nextState, nextReward, nextTerminated, _, _ = env.step(action)

        agent.update(state, action, reward, nextState, nextAction)
        
        action = nextAction
        state = nextState
        reward = nextReward
        terminated = nextTerminated

        if render_on:
            env.render()
            time.sleep(0.05)

    return agent.e_table

def plot_heatmap(e_table, lambda_val):
    """
    Plots the eligibility trace table as a heatmap.
    """

    # ---- reshape to (5,5,4) so we can pick the *best* action per cell ----
    e_grid = e_table.reshape(5, 5, 4)               # (row, col, action)

    # ---- pick the action with the largest trace (or 0 if all zero) ----
    best_e   = np.max(e_grid, axis=2)               # (5,5)  max trace per cell
    best_act = np.argmax(e_grid, axis=2)            # (5,5)  which action gave it

    # ---- create the figure ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim(0, 5)
    # ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(range(6))
    ax.set_yticklabels(range(6).__reversed__())   # row 0 at the top
    ax.grid(True, linewidth=1.5, color='k')

    # ---- colour each cell by its max trace --------------------------------
    norm = plt.Normalize(vmin=0, vmax=best_e.max() or 1)
    cmap = plt.cm.hot

    for row in range(5):
        for col in range(5):
            val = best_e[row, col]
            if val == 0:                     # nothing to show
                continue

            # # background colour
            # ax.add_patch(plt.Rectangle((col, 4-row), 1, 1,
            #              facecolor=cmap(norm(val)), edgecolor='k'))

            # text: the trace value (rounded) + optional action arrow
            txt = f"{val:.2f}"
            ax.text(col+0.5, 4-row+0.5, txt,
                    ha='center', va='center', fontsize=9, weight='bold')

    ax.set_title(f'Eligibility-trace "credit trail" (λ = {lambda_val})')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    plt.savefig(f'heatmap_lambda_{lambda_val}.png', dpi=200)
    plt.close()

# --- Main script execution ---
if __name__ == "__main__":
    env = MazeEnv()

    # Set this to True to watch the agent explore, False to run quickly
    VISUALIZE_EXPLORATION = False
    
    # Lambda values to test as required by the homework
    lambda_values = [0.2, 0.7, 0.95]
    
    for l_val in lambda_values:
        # Create a new agent for each lambda value to start fresh
        agent = SarsaLambdaAgent(
            action_space_size=env.action_space.n,
            state_space_size=env.observation_space.n,
            lambda_val=l_val
        )

        # Pass the visualization flag to the function
        final_e_table = find_successful_episode(agent, env, render_on=VISUALIZE_EXPLORATION)
        
        # Plot the heatmap of the traces
        plot_heatmap(final_e_table, l_val)
        
    env.close()