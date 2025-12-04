import mujoco
import mujoco.viewer
import time
import numpy as np
import ompl.base as ob
import ompl.geometric as og
from scipy.optimize import minimize
from stable_baselines3 import PPO


def ik(model, data, targetPos, initialQpos=None, tol=1e-4, maxIter=100, alpha=0.1):
    siteId = model.site("endEffector").id
    numJoints = model.nq

    if initialQpos is None:
        initialQpos = data.qpos.copy()

    bounds = []
    for i in range(numJoints):
        if model.jnt_limited[i]:
            bounds.append((model.jnt_range[i][0], model.jnt_range[i][1]))
        else:
            bounds.append((-10 * np.pi, 10 * np.pi))

    def objective(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        currentPos = data.site(siteId).xpos
        posError = np.linalg.norm(currentPos - targetPos)
        regError = alpha * np.linalg.norm(q - initialQpos)
        return posError**2 + regError**2

    res = minimize(
        objective,
        initialQpos,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": maxIter, "ftol": tol},
    )

    print(res)

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
        <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <geom name="obstacle" type="box" pos="0.45 0.25 0.55" size="0.3 0.1 0.025" rgba="1 0.5 0 1" />
        <body name="base" pos="0 0 0">
            <geom name="baseBox" type="box" size="0.1 0.1 0.05"/>
        """
        currentPos = "0 0 0.05"
        numCloses = 0
        colors = [
            "0.8 0.2 0.2 1",
            "0.2 0.8 0.2 1",
            "0.2 0.2 0.8 1",
            "0.8 0.8 0.2 1",
            "0.2 0.8 0.8 1",
            "0.8 0.2 0.8 1",
            "0.5 0.5 0.5 1",
        ]
        for i in range(numJoints):
            color = colors[i % len(colors)]
            if jointTypes[i] == 0:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0" rgba="{color}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 1:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0" rgba="{color}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 2:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 0 1" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0" rgba="{color}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            else:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <geom name="baseCapsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0" rgba="{color}"/>
                    <body name="slideChild{i}"> 
                        <joint name="joint{i}" type="slide" axis="0 0 1" range="0 {lengths[i]}" damping="1.0"/>
                        <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0" rgba="{color}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 2
        xml += (
            f'<site name="endEffector" pos="{currentPos}" size="0.01" rgba="0 1 0 1"/>'
        )
        xml += "</body>" * numCloses  # Close links
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


# Load PPO model and get configuration
model_path = "shelfArm"
print(f"Loading PPO model from {model_path}...")
ppo_model = PPO.load(model_path)

# Get action from policy (deterministic for best performance)
obs = np.array([0.0], dtype=np.float32)
action, _ = ppo_model.predict(obs, deterministic=True)

# Decode action
# Constants from robotArmEnv
MIN_NUM_LINKS = 2
MAX_NUM_LINKS = 7
MIN_LENGTH = 0.05
MAX_LENGTH = 1.2

# Decode number of links (Force to 4 as requested)
numLinks = 4

# Decode lengths
# action[1:8] corresponds to lengths
raw_lengths = action[1 : (MAX_NUM_LINKS + 1)]
decoded_lengths = (raw_lengths * (MAX_LENGTH - MIN_LENGTH) + MIN_LENGTH)[:numLinks]
lengths = decoded_lengths.tolist()

# Decode joint types
# action[8:15] corresponds to joint types
raw_joint_types = action[(1 + MAX_NUM_LINKS) :]
decoded_joint_types = np.round(raw_joint_types * 3)[:numLinks].astype(int)
jointTypes = decoded_joint_types.tolist()

print(f"PPO Config:")
print(f"  Links: {numLinks}")
print(f"  Lengths: {np.round(lengths, 4)}")
print(f"  Joint Types: {jointTypes}")

xml = generateXML(numLinks, lengths, jointTypes)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

actuatorIds = [model.actuator(f"motor{i}").id for i in range(numLinks)]
jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
obstacleId = model.geom("obstacle").id

space = ob.CompoundStateSpace()

isSO2 = []

for link in range(numLinks):
    if jointTypes[link] == 2:
        space.addSubspace(ob.SO2StateSpace(), 1.0)
        isSO2.append(True)
    elif jointTypes[link] == 3:
        subspace = ob.RealVectorStateSpace(1)
        space.addSubspace(subspace, 1.0)
        bounds = ob.RealVectorBounds(1)
        bounds.setLow(0, 0)
        bounds.setHigh(0, lengths[link])
        subspace.setBounds(bounds)
        isSO2.append(False)
    else:
        subspace = ob.RealVectorStateSpace(1)
        space.addSubspace(subspace, 1.0)
        bounds = ob.RealVectorBounds(1)
        bounds.setLow(0, -np.pi / 2)
        bounds.setHigh(0, np.pi / 2)
        subspace.setBounds(bounds)
        isSO2.append(False)


def isStateValid(state):
    qpos = np.zeros(numLinks)
    for i in range(numLinks):
        if isSO2[i]:
            qpos[i] = state[i].value
        else:
            qpos[i] = state[i][0]

    if not np.all(np.isfinite(qpos)):
        return False

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for j in range(data.ncon):
        contact = data.contact[j]
        if contact.geom1 == obstacleId or contact.geom2 == obstacleId:
            return False
    return True


validityChecker = ob.StateValidityCheckerFn(isStateValid)
si = ob.SpaceInformation(space)
si.setStateValidityChecker(validityChecker)
simpleSetup = og.SimpleSetup(si)

startPos = np.array([0.41, 0.21, 0.3])
goalPos = np.array([0.4, 0.2, 0.8])

startQpos = ik(model, data, startPos)
print(f"Starting angles: {startQpos}")
goalQpos = ik(model, data, goalPos, initialQpos=startQpos)

i = 0
for id in jointIds:
    data.qpos[id] = goalQpos[i]
    i += 1

mujoco.mj_forward(model, data)
goalError = np.linalg.norm(data.site("endEffector").xpos - goalPos)
print(f"Goal error is: {goalError}")

i = 0
for id in jointIds:
    data.qpos[id] = startQpos[i]
    i += 1

mujoco.mj_forward(model, data)
startError = np.linalg.norm(data.site("endEffector").xpos - startPos)
print(f"Start error is: {startError}")

start = ob.State(space)
goal = ob.State(space)
for i in range(len(startQpos)):
    if isSO2[i]:
        start()[i].value = startQpos[i]
        goal()[i].value = goalQpos[i]
    else:
        start()[i][0] = startQpos[i]
        goal()[i][0] = goalQpos[i]

simpleSetup.setStartAndGoalStates(start, goal)

planner = og.RRTstar(si)
simpleSetup.setPlanner(planner)
simpleSetup.solve(10.0)

if simpleSetup.haveSolutionPath():
    simpleSetup.simplifySolution()
    path = simpleSetup.getSolutionPath()
    path.interpolate(100)
    pathStates = [path.getState(i) for i in range(path.getStateCount())]
    print(f"Found path with {len(pathStates)} states.")
else:
    print("No path found")
    pathStates = []

index = 0

model.site("startPos").pos = startPos
model.site("goalPos").pos = goalPos

viewer = mujoco.viewer.launch_passive(model, data)

viewer.cam.lookat[:] = model.stat.center
viewer.cam.distance = model.stat.extent * 2
viewer.cam.elevation = -35
viewer.cam.azimuth = 145

mujoco.mj_forward(model, data)
viewer.sync()

input("Press enter to continue...")

index = 0
while viewer.is_running() and index < len(pathStates):
    for i, jid in enumerate(jointIds):
        if not isSO2[i]:
            print(pathStates[index][i][0])
        data.qpos[jid] = (
            pathStates[index][i].value if isSO2[i] else pathStates[index][i][0]
        )

    mujoco.mj_forward(model, data)
    viewer.sync()

    time.sleep(0.05)
    index += 1

print("Sim Complete")

while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)
