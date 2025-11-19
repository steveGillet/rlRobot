import mujoco
import mujoco.viewer
import time
import numpy as np
import ompl.base as ob
import ompl.geometric as og
from scipy.optimize import minimize
import keyboard

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
    
    res = minimize(objective, initialQpos, bounds=bounds, method='L-BFGS-B', options={'maxiter': maxIter, 'ftol': tol})

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

numLinks = 3
xml = generateXML(numLinks, [0.3, 0.2, 0.3], [2, 1, 3])
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

actuatorIds = [model.actuator(f"motor{i}").id for i in range(numLinks)]
jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]

space = ob.CompoundStateSpace()
for link in range(numLinks):
    space.addSubspace(ob.SO2StateSpace(), 1.0)

si = ob.SpaceInformation(space)
simpleSetup = og.SimpleSetup(si)

startPos = np.array([-.4, -0.4, 0.6])
goalPos = np.array([0.4, 0.4, 0.8])

startQpos = ik(model, data, startPos)
print(f"Starting angles: {startQpos}")
goalQpos = ik(model, data, goalPos, initialQpos=startQpos)

i = 0
for id in jointIds:
    data.qpos[id] = startQpos[i]
    i+=1

mujoco.mj_forward(model, data)
startError = np.linalg.norm(data.site('endEffector').xpos - startPos)
print(f"Start error is: {startError}")

start = ob.State(space)
goal = ob.State(space)
for i in range(len(startQpos)):
    start()[i].value = startQpos[i]
    goal()[i].value = goalQpos[i]
    
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
    print('No path found')
    pathStates = []

index = 0

model.site('startPos').pos = startPos
model.site('goalPos').pos = goalPos

viewer = mujoco.viewer.launch_passive(model, data)

viewer.cam.lookat[:] = model.stat.center
viewer.cam.distance = model.stat.extent * 2
viewer.cam.elevation = -35
viewer.cam.azimuth = 145

mujoco.mj_forward(model,data)
viewer.sync()

keyboard.wait('space')

index = 0
while viewer.is_running() and index < len(pathStates):
    for i, jid in enumerate(jointIds):
        data.qpos[jid] = pathStates[index][i].value

    mujoco.mj_forward(model, data)
    viewer.sync()

    time.sleep(0.05)
    index += 1

print("Sim Complete")

while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)
    