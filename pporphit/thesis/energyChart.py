import matplotlib.pyplot as plt
import numpy as np

# Categories
tasks = ['Container', 'Outreach', 'Shelf', 'Side-to-Side', 'Wall Mount']
models = ['PPO', 'SAC', 'CMA-ES', 'FANUC', 'Panda']

# Energy Data (Joules)
# We use 0 to represent infinity/failure. The log scale handles this gracefully by dropping the bar.
energy_ppo = [0, 0, 0, 23.50, 0]
energy_sac = [31.87, 24.22, 100.36, 23.49, 37.29]
energy_cma = [31.40, 46.29, 265.16, 31.31, 46.40]
energy_fanuc = [0, 0, 0, 0, 0]
energy_panda = [36898.65, 337.13, 10718.82, 4669.10, 257456.71]

x = np.arange(len(tasks))
width = 0.15 

colors = ['#d62728', '#ff7f0e', '#1f77b4', '#7f7f7f', '#2ca02c']

fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the bars. We use log=True to handle the massive gap between PMorph and Panda.
# matplotlib will automatically ignore the 0s (failures) on a log scale and just won't draw a bar.
ax.bar(x - 2*width, energy_ppo, width, label='PPO', color=colors[0], log=True)
ax.bar(x - width, energy_sac, width, label='SAC', color=colors[1], log=True)
ax.bar(x, energy_cma, width, label='CMA-ES', color=colors[2], log=True)
ax.bar(x + width, energy_fanuc, width, label='FANUC', color=colors[3], log=True)
ax.bar(x + 2*width, energy_panda, width, label='Panda', color=colors[4], log=True)

# Formatting
ax.set_ylabel('Energy Consumption (Joules) - Log Scale', fontsize=14)
ax.set_title('Task Energy Efficiency (Logarithmic Scale)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12)

# Add a note explaining the missing bars
plt.figtext(0.15, 0.02, "Note: Missing bars indicate task failure (\u221E energy).", 
            ha="left", fontsize=10, style='italic', color='dimgray')

# Grid and Legend
ax.grid(axis='y', linestyle='--', alpha=0.7, which='both') # 'both' gets the minor log ticks too
ax.legend(fontsize=12, loc='upper left')

plt.tight_layout()
# Adjust bottom margin to make room for the note
plt.subplots_adjust(bottom=0.15) 

plt.savefig('Energy_Consumption_Chart_LogScale.png', dpi=300)
plt.show()