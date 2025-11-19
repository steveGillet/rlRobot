import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# Simple RRT implementation (lightweight)
class SimpleRRT:
    def __init__(self, model, data, num_joints, start_q, goal_q):
        self.model = model
        self.data = data
        self.num_joints = num_joints
        self.start_q = start_q
        self.goal_q = goal_q
        self.tree = [start_q]
        self.max_iter = 1000  # Increased for better success

    def _sample(self):
        return np.random.uniform(-np.pi/4, np.pi/4, self.num_joints)  # Tighter for stability

    def _collision_free(self, q):
        self.data.qpos[:self.num_joints] = q
        mujoco.mj_step(self.model, self.data)
        return self.data.ncon == 0  # No contacts = free

    def _near_goal(self, q):
        return np.linalg.norm(q - self.goal_q) < 0.1

    def plan(self):
        for _ in range(self.max_iter):
            q_rand = self._sample()
            q_near = self.tree[np.argmin([np.linalg.norm(q_rand - q) for q in self.tree])]
            q_new = q_near + 0.1 * (q_rand - q_near) / np.linalg.norm(q_rand - q_near)
            if self._collision_free(q_new):
                self.tree.append(q_new)
                if self._near_goal(q_new):
                    return True
        return False

class ManipulatorEnv(gym.Env):
    def __init__(self, max_joints=7):
        super().__init__()
        self.max_joints = max_joints
        self.action_space = spaces.Box(low=0, high=1, shape=(1 + max_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.model = None
        self.data = None

    def _generate_mjcf(self, num_joints, lengths):
        try:
            xml = """
<mujoco model="manipulator">
    <compiler angle="radian"/>
    <option timestep="0.005" tolerance="1e-10" impratio="10" iterations="1000"/>
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.05 0.05 0.05" mass="0.1"/>
            """
            current_pos = "0 0 0"
            for i in range(num_joints):
                xml += f"""
                <body name="link{i}" pos="{current_pos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 {lengths[i]} 0 0" mass="0.1"/>
                """
                current_pos = f"{lengths[i]} 0 0"
            xml += "</body>" * num_joints  # Close links
            xml += """
        </body>  <!-- Close base -->
        <body name="box" pos="0.5 0.5 0.1">
            <freejoint name="box_joint"/>
            <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="0.5"/>
        </body>
        <body name="goal" pos="1.0 1.0 0.1">
            <geom name="goal_geom" type="sphere" size="0.05" rgba="0 1 0 0.5"/>
        </body>
</worldbody>
<actuator>
            """
            for i in range(num_joints):
                xml += f'<motor name="motor{i}" joint="joint{i}"/>'
            xml += """
</actuator>
</mujoco>
            """
            return xml
        except Exception as e:
            print(f"MJCF Generation Error: {e}")
            raise

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.box_pos = np.random.uniform([-0.5, -0.5, 0.1], [0.5, 0.5, 0.1])
        self.goal_pos = np.random.uniform([0.5, 0.5, 0.1], [1.0, 1.0, 0.1])
        obs = np.concatenate([self.box_pos, self.goal_pos])
        info = {}
        return obs, info

    def step(self, action):
        num_joints = int(3 + action[0] * 4)  # 3 to 7
        lengths = action[1:1+num_joints] * 0.9 + 0.1  # [0.1, 1.0]
        lengths = np.pad(lengths, (0, self.max_joints - num_joints), constant_values=0.5)[:num_joints]

        try:
            xml = self._generate_mjcf(num_joints, lengths)
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
        except ValueError as e:
            print(f"XML Load Error: {e}\nXML:\n{xml}")
            return np.zeros(6), -10.0, True, False, {}

        # Set positions
        box_geom_id = self.model.geom('box_geom').id
        goal_geom_id = self.model.geom('goal_geom').id
        self.data.geom_xpos[box_geom_id] = self.box_pos
        self.data.geom_xpos[goal_geom_id] = self.goal_pos

        # RRT: Plan to push config
        start_q = np.zeros(num_joints)
        goal_q = np.random.uniform(-np.pi/2, np.pi/2, num_joints)  # Placeholder
        rrt = SimpleRRT(self.model, self.data, num_joints, start_q, goal_q)
        success = rrt.plan()
        print("RRT success:", success)  # Debug

        # Simulate push (apply force to end-effector if success)
        if success:
            self.data.qpos[:num_joints] = np.clip(goal_q, -np.pi/3, np.pi/3)  # Safer range
            ee_body_id = self.model.body(f"link{num_joints-1}").id
            for _ in range(50):
                self.data.xfrc_applied[ee_body_id] = np.array([0.1, 0, 0, 0, 0, 0])  # 5x smaller
                mujoco.mj_step(self.model, self.data)
                if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
                    return np.zeros(6), -100.0, True, False, {"unstable": True}  # Penalty
            # Clip velocities too
            self.data.qvel[:num_joints] = np.clip(self.data.qvel[:num_joints], -10, 10)
            self.box_pos = self.data.geom_xpos[box_geom_id]

        dist = np.linalg.norm(self.box_pos - self.goal_pos)
        reward = 1.0 if dist < 0.1 else -dist
        terminated = dist < 0.1
        truncated = False
        obs = np.concatenate([self.box_pos, self.goal_pos])
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.model is not None:
            mujoco.viewer.launch_passive(self.model, self.data).sync()