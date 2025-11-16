import mujoco
import mujoco.viewer
import time
import numpy as np
import ompl.base as ob
import ompl.geometric as og
from scipy.optimize import minimize
import gymnasium as gym
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from ompl import util as ou
ou.setLogLevel(ou.LOG_NONE)

class robotArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, numLinks=4, fixedJointTypes=None):
        super().__init__()
        self.numLinks = numLinks
        self.action_space = gym.spaces.Box(low=0.05, high=1.2, shape=(numLinks,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

        self.fixedJointTypes = fixedJointTypes or [2,0,2,1]
        self.startPos = np.array([-.4, -0.4, 0.6], dtype=np.float32)
        self.goalPos = np.array([0.4, 0.4, 0.8], dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.array([0.0], dtype=np.float32), {}

    def _evaluate(self, lengths):
        try:
            xml = generateXML(self.numLinks, lengths.tolist(), self.fixedJointTypes)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
        except:
            return -50.0

        actuatorIds = [model.actuator(f"motor{i}").id for i in range(self.numLinks)]
        jointIds = [model.joint(f"joint{i}").id for i in range(self.numLinks)]

        space = ob.CompoundStateSpace()
        for link in range(self.numLinks):
            space.addSubspace(ob.SO2StateSpace(), 1.0)

        si = ob.SpaceInformation(space)
        simpleSetup = og.SimpleSetup(si)

        startQpos = ik(model, data, self.startPos)
        goalQpos = ik(model, data, self.goalPos, initialQpos=startQpos)
        
        if startQpos is None:
            return -100.0

        if goalQpos is None:
            return -100.0


        i = 0
        for id in jointIds:
            data.qpos[id] = goalQpos[i]
            i+=1

        mujoco.mj_forward(model, data)
        goalError = np.linalg.norm(data.site('endEffector').xpos - self.goalPos)

        i = 0
        for id in jointIds:
            data.qpos[id] = startQpos[i]
            i+=1

        mujoco.mj_forward(model, data)
        startError = np.linalg.norm(data.site('endEffector').xpos - self.startPos)

        start = ob.State(space)
        goal = ob.State(space)
        for i in range(len(startQpos)):
            start()[i].value = startQpos[i]
            goal()[i].value = goalQpos[i]
            
        simpleSetup.setStartAndGoalStates(start, goal)

        planner = og.RRTstar(si)
        simpleSetup.setPlanner(planner)
        simpleSetup.solve(8.0)

        if simpleSetup.haveSolutionPath():
            simpleSetup.simplifySolution()
            path = simpleSetup.getSolutionPath()
            path.interpolate(20)
            # pathStates = [path.getState(i) for i in range(path.getStateCount())]
            return 100 - 0.8 * path.length() - 20 * (startError + goalError)

        else:
            # pathStates = []
            return 30 - 50 * (startError + goalError)
        
    def step(self, action):
        lengths = np.clip(action, 0.05, 1.2)
        reward = self._evaluate(lengths)
        done = True
        return np.array([0.0], dtype=np.float32), reward, done, done, {}

def ik(model, data, targetPos, initialQpos=None, tol=1e-4, maxIter=100, alpha=0.1):
    siteId = model.site('endEffector').id
    numJoints = model.nq

    if initialQpos is None:
        initialQpos = data.qpos.copy()

    bounds = []
    for i in range(numJoints):
        if model.jnt_limited[i]:
            bounds.append((model.jnt_range[i][0], model.jnt_range[i][1]))
        else:
            bounds.append((-10*np.pi, 10*np.pi))

    def objective(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        currentPos = data.site(siteId).xpos
        posError = np.linalg.norm(currentPos - targetPos)
        regError = alpha * np.linalg.norm(q - initialQpos)
        return posError**2 + regError**2
    
    res = minimize(objective, initialQpos, bounds=bounds, method='L-BFGS-B', options={'maxiter': maxIter, 'ftol': tol, 'disp': False})

    # print(res)

    if res.success:
        return res.x
    else:
        print(f"IK failed: {res.message}")
        return None
    
def generateXML(numJoints, lengths, jointTypes):
    try:
        xml = """
<mujoco>
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.05"/>
        """
        currentPos = "0 0 0.05"
        for i in range(numJoints):
            if jointTypes[i] == 0:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="1.0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"
            elif jointTypes[i] == 1:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"
            elif jointTypes[i] == 2:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 0 1" damping="1.0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"
            else:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="slide" axis="0 0 1" pos="{currentPos}" range="0 1" damping="1.0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"    
        xml += f'<site name="endEffector" pos="{currentPos}" size="0.01" rgba="0 1 0 1"/>'
        xml += "</body>" * numJoints  # Close links
        xml += """
        </body>  <!-- Close base -->
    <site name="startPos" pos="0 1 -1" size="0.02" rgba="0 0 1 1"/>
    <site name="goalPos" pos="-2 0 -1" size="0.02" rgba="1 0 0 1"/>
  </worldbody>
<actuator>
        """
        for i in range(numJoints):
            xml += f'<motor name="motor{i}" joint="joint{i}" ctrlrange="-10 10"/>'
        xml += """
</actuator>
</mujoco>
        """
        return xml
    except Exception as e:
        print(f"Mujoco XML Generation Error: {e}")
        raise

def makeEnv():
    def _init():
        return robotArmEnv(4)
    return _init

# if __name__ == '__main__':
#     venv = DummyVecEnv([makeEnv() for _ in range(4)])
#     venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

#     policyKwargs = dict(net_arch=[128,128,128])
#     ppo = PPO("MlpPolicy", venv, policy_kwargs=policyKwargs, learning_rate=0.001, n_steps=128, batch_size=512, n_epochs=4, gamma=0.98, verbose=1, tensorboard_log="./arm_morph_tb/", device="cpu")
#     ppo.learn(total_timesteps=3_000_000)
#     ppo.save("bestestArm")

if __name__ == "__main__":
    env = robotArmEnv(4)
    obs, info = env.reset()
    for _ in range(5):
        a = env.action_space.sample()
        print("step...")
        obs, r, done, trunc, info = env.step(a)
        print("reward:", r)


# index = 0

# model.site('startPos').pos = startPos
# model.site('goalPos').pos = goalPos

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while index < len(pathStates):
#         i = 0
#         for id in jointIds:
#             data.qpos[id] = pathStates[index][i].value
#             i+=1

#         print(data.qpos[2])

#         mujoco.mj_step(model,data)
#         print(f"Time step: {data.time}s, Position: {data.geom_xpos[-1]}")
#         viewer.sync()
#         time.sleep(0.1)
#         index += 1

# print("Sim Complete") 