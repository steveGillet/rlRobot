import mujoco
import mujoco.viewer
import time
import numpy as np
import ompl.base as ob
import ompl.geometric as og
from scipy.optimize import minimize

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

model = mujoco.MjModel.from_xml_path('arm.xml')
data = mujoco.MjData(model)
shoulderMotorId = model.actuator('shoulderMotor').id
elbowMotorId = model.actuator('elbowMotor').id
shoulderId = model.joint('shoulder').id
elbowId = model.joint('elbow').id


space = ob.CompoundStateSpace()
space.addSubspace(ob.SO2StateSpace(), 1.0)
space.addSubspace(ob.SO2StateSpace(), 1.0)

si = ob.SpaceInformation(space)
simpleSetup = og.SimpleSetup(si)

startPos = np.array([-.2, -0.2, 0.3])
goalPos = np.array([0.2, 0.2, 0.4])

startQpos = ik(model, data, startPos)
print(f"Starting angles: {startQpos}")
goalQpos = ik(model, data, goalPos, initialQpos=startQpos)

data.qpos[elbowId] = startQpos[0]
data.qpos[shoulderId] = startQpos[1]
mujoco.mj_forward(model, data)
startError = np.linalg.norm(data.site('endEffector').xpos - startPos)
print(f"Start error is: {startError}")

start = ob.State(space)
start()[0].value = startQpos[0]
start()[1].value = startQpos[1]
goal = ob.State(space)
goal()[0].value = goalQpos[0]
goal()[1].value = goalQpos[1]
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

with mujoco.viewer.launch_passive(model, data) as viewer:
    while index < len(pathStates):
        print(pathStates[index][0].value)
        data.qpos[shoulderId] = pathStates[index][0].value
        data.qpos[elbowId] = pathStates[index][1].value

        mujoco.mj_step(model,data)
        print(f"Time step: {data.time}s, Position: {data.geom_xpos[-1]}")
        viewer.sync()
        time.sleep(0.1)
        index += 1

print("Sim Complete")