import matplotlib.pyplot as plt
import numpy as np

# 1. Categories
tasks = ['Container', 'Outreach', 'Shelf', 'Side-to-Side', 'Wall Mount']
models = ['PPO', 'SAC', 'CMA-ES', 'FANUC', 'Panda']

# 2. Extracted Data from LaTeX Table
# Success Rates
success_ppo = [0.0, 0.0, 0.0, 100.0, 0.0]
success_sac = [21.5, 92.0, 5.0, 100.0, 50.0]
success_cma = [3.5, 100.0, 74.5, 100.0, 50.0]
success_fanuc = [0.0, 0.0, 0.0, 0.0, 0.0]
success_panda = [55.0, 100.0, 100.0, 100.0, 37.0]

# Number of Links
links_ppo = [7, 2, 2, 2, 2]
links_sac = [2, 2, 2, 2, 2]
links_cma = [2, 4, 4, 3, 2]
links_fanuc = [6, 6, 6, 6, 6]
links_panda = [7, 7, 7, 7, 7]

x = np.arange(len(tasks))
width = 0.15 # Adjusted width for 5 bars

# Plotting style adjustments for a professional look
colors = ['#d62728', '#ff7f0e', '#1f77b4', '#7f7f7f', '#2ca02c']

# --- PLOT 1: SUCCESS RATE ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x - 2*width, success_ppo, width, label='PPO', color=colors[0])
ax1.bar(x - width, success_sac, width, label='SAC', color=colors[1])
ax1.bar(x, success_cma, width, label='CMA-ES', color=colors[2])
ax1.bar(x + width, success_fanuc, width, label='FANUC', color=colors[3])
ax1.bar(x + 2*width, success_panda, width, label='Panda', color=colors[4])

ax1.set_ylabel('Success Rate (%)', fontsize=14)
ax1.set_title('Task Success Rate Comparison', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, fontsize=12)
ax1.legend(fontsize=12, loc='upper left')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Success_Rate_Chart.png', dpi=300)
plt.show()

# --- PLOT 2: NUMBER OF LINKS ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.bar(x - 2*width, links_ppo, width, label='PPO', color=colors[0])
ax2.bar(x - width, links_sac, width, label='SAC', color=colors[1])
ax2.bar(x, links_cma, width, label='CMA-ES', color=colors[2])
ax2.bar(x + width, links_fanuc, width, label='FANUC', color=colors[3])
ax2.bar(x + 2*width, links_panda, width, label='Panda', color=colors[4])

ax2.set_ylabel('Number of Links', fontsize=14)
ax2.set_title('Optimized Morphology vs Baselines', fontsize=16, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(tasks, fontsize=12)
ax2.legend(fontsize=12, loc='upper right')
ax2.set_ylim(0, 8)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Number_of_Links_Chart.png', dpi=300)
plt.show()