import mujoco
import numpy as np

class Node:
    def __init__(self, config):
        self.config = config
        self.parent = None

def is_collision_free(model, data, config):
    data.qpos[:] = config
    mujoco.mj_forward(model, data)
    return data.ncon == 0  # Adjust based on specific collision criteria

def sample_config(joint_limits):
    return np.random.uniform(joint_limits[0], joint_limits[1])

def nearest_node(tree, config):
    dists = [np.linalg.norm(node.config - config) for node in tree]
    return tree[np.argmin(dists)]

def extend(from_config, to_config, step_size):
    direction = to_config - from_config
    dist = np.linalg.norm(direction)
    if dist <= step_size:
        return to_config
    else:
        return from_config + (direction / dist) * step_size

def rrt(model, start_config, goal_config, joint_limits, max_iter=1000, step_size=0.1, goal_bias=0.05):
    tree = [Node(start_config)]
    planning_data = mujoco.MjData(model)
    for _ in range(max_iter):
        if np.random.rand() < goal_bias:
            rand_config = goal_config
        else:
            rand_config = sample_config(joint_limits)
        nearest = nearest_node(tree, rand_config)
        new_config = extend(nearest.config, rand_config, step_size)
        if is_collision_free(model, planning_data, new_config):
            new_node = Node(new_config)
            new_node.parent = nearest
            tree.append(new_node)
            if np.linalg.norm(new_config - goal_config) < step_size:
                path = []
                current = new_node
                while current:
                    path.append(current.config)
                    current = current.parent
                return path[::-1]
    return None