import numpy as np

# Grid setup: 4x4, row 0 top, col 0 left
rows, cols = 4, 4
gamma = 0.9  # Discount factor
step_reward = -1  # w = -1 by default
goal_reward = 50
trap_reward = -50
goal = (0, 3)  # G
trap = (1, 3)  # T
obstacle = (1, 1)  # O
wormhole_entry = (1, 2)  # W
wormhole_exit = (3, 3)  # E
terminals = {goal, trap}

# Actions: 0: North, 1: South, 2: East, 3: West
actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
action_symbols = ['\\uparrow', '\\downarrow', '\\rightarrow', '\\leftarrow']  # LaTeX symbols for policy grid

# Initialize V to 0
V = np.zeros((rows, cols))

# Value Iteration
threshold = 1e-6
iteration = 0
while True:
    iteration += 1
    delta = 0
    new_V = np.copy(V)
    for i in range(rows):
        for j in range(cols):
            if (i, j) in terminals or (i, j) == obstacle:
                continue  # Skip terminals and obstacle
            qs = []
            for a in range(4):
                di, dj = actions[a]
                ni, nj = i + di, j + dj
                if not (0 <= ni < rows and 0 <= nj < cols) or (ni, nj) == obstacle:
                    next_state = (i, j)
                elif (ni, nj) == wormhole_entry:
                    next_state = wormhole_exit
                else:
                    next_state = (ni, nj)
                
                if next_state == goal:
                    r = goal_reward
                elif next_state == trap:
                    r = trap_reward
                else:
                    r = step_reward
                
                q = r + gamma * V[next_state[0], next_state[1]]
                qs.append(q)
            new_V[i, j] = max(qs)
            delta = max(delta, abs(new_V[i, j] - V[i, j]))
    V = new_V
    if delta < threshold:
        print(f"Converged after {iteration} iterations.")
        break

# Extract policy
policy = np.full((rows, cols), '', dtype=object)
for i in range(rows):
    for j in range(cols):
        if (i, j) in terminals or (i, j) == obstacle or (i, j) == wormhole_entry:
            continue
        qs = []
        for a in range(4):
            di, dj = actions[a]
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols) or (ni, nj) == obstacle:
                next_state = (i, j)
            elif (ni, nj) == wormhole_entry:
                next_state = wormhole_exit
            else:
                next_state = (ni, nj)
            
            if next_state == goal:
                r = goal_reward
            elif next_state == trap:
                r = trap_reward
            else:
                r = step_reward
            
            q = r + gamma * V[next_state[0], next_state[1]]
            qs.append(q)
        best_a = np.argmax(qs)  # In case of ties, picks the first (e.g., North over West)
        policy[i, j] = action_symbols[best_a]

# Print V grid (rounded to 2 decimals for display; use full precision in LaTeX if needed)
print("Optimal Values V*:")
for i in range(rows):
    row_str = []
    for j in range(cols):
        if (i, j) in terminals:
            row_str.append("0.00")
        elif (i, j) == obstacle or (i, j) == wormhole_entry:
            row_str.append("N/A")
        else:
            row_str.append(f"{V[i, j]:.2f}")
    print(" | ".join(row_str))

# Print policy grid (with LaTeX symbols)
print("\nOptimal Policy (LaTeX symbols):")
for i in range(rows):
    row_str = []
    for j in range(cols):
        if (i, j) == goal:
            row_str.append("G")
        elif (i, j) == trap:
            row_str.append("T")
        elif (i, j) == obstacle:
            row_str.append("O")
        elif (i, j) == wormhole_entry:
            row_str.append("W")
        else:
            row_str.append(policy[i, j])
    print(" & ".join(row_str) + " \\\\")

# Output full precision values for reference
print("\nFull Precision Values:")
print(V)