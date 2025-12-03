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
    
def ik_dls(
    model,
    target_pos: np.ndarray,
    initialQpos: np.ndarray | None = None,
    max_iters: int = 200,
    tol: float = 1e-3,
    lambda_: float = 1e-2,
    max_step: float = 0.3,
) -> np.ndarray | None:
    """
    Damped least-squares IK for site 'endEffector'.

    Returns:
        q_best (np.ndarray of shape (nq,)) or None if something is badly wrong
        (NaNs, singular beyond recovery, etc.).
    """
    site_id = model.site("endEffector").id
    nq = model.nq
    nv = model.nv  # for your chain, nv == nq

    # Separate data object so IK can't corrupt caller's MjData
    ik_data = mujoco.MjData(model)

    # --- Initial guess ---
    if initialQpos is None:
        q = np.zeros(nq, dtype=np.float64)
    else:
        q = np.array(initialQpos, dtype=np.float64).copy()
        if q.shape[0] != nq or not np.all(np.isfinite(q)):
            q = np.zeros(nq, dtype=np.float64)

    # --- Joint limits in q-space ---
    q_min = np.full(nq, -np.inf)
    q_max = np.full(nq,  np.inf)
    for j in range(model.njnt):
        adr = model.jnt_qposadr[j]  # index of this joint in qpos
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            q_min[adr] = lo
            q_max[adr] = hi

    q = np.clip(q, q_min, q_max)

    # Track best pose seen
    q_best = q.copy()
    err_best = np.inf

    # Jacobian buffers
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for _ in range(max_iters):
        # Sanity check
        if not np.all(np.isfinite(q)):
            return None, None

        # Forward kinematics
        ik_data.qpos[:] = q
        mujoco.mj_forward(model, ik_data)
        current_pos = np.array(ik_data.site(site_id).xpos, copy=True)

        err = target_pos - current_pos
        err_norm = float(np.linalg.norm(err))

        # Track best
        if np.isfinite(err_norm) and err_norm < err_best:
            err_best = err_norm
            q_best = q.copy()

        # Good enough?
        if err_norm < tol:
            break

        # Compute Jacobian
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = jacp[:, :nv]  # (3, nv)

        # Damped least-squares step
        A = J @ J.T + (lambda_ ** 2) * np.eye(3)
        try:
            v = np.linalg.solve(A, err)        # (3,)
        except np.linalg.LinAlgError:
            # Very ill-conditioned; use best so far
            break

        dq = J.T @ v  # (nv,)

        # Limit step size to avoid crazy jumps
        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step:
            dq *= max_step / (step_norm + 1e-8)

        q = q + dq
        q = np.minimum(np.maximum(q, q_min), q_max)

    # Final sanity check
    if not np.all(np.isfinite(q_best)):
        return None, None
    
    try:
        ik_data.qpos[:] = q_best
        mujoco.mj_forward(model, ik_data)
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        jBest = jacp[:, :nv]
        if not np.all(np.isfinite(jBest)):
            jBest = np.zeros((3, nv))  # Fallback for non-finite Jacobian
    except Exception as e:
        print(f"jBest computation failed: {e}")
        jBest = np.zeros((3, nv))
    return q_best, jBest
    
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
        for i in range(numJoints):
            if jointTypes[i] == 0:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses +=1
            elif jointTypes[i] == 1:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 2:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 0 1" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            else:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <geom name="baseCapsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                    <body name="slideChild{i}"> 
                        <joint name="joint{i}" type="slide" axis="0 0 1" range="0 {lengths[i]}" damping="1.0"/>
                        <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="1.0"/>
                """
                currentPos = f"0 0 {lengths[i]}"    
                numCloses += 2
        xml += f'<site name="endEffector" pos="{currentPos}" size="0.01" rgba="0 1 0 1"/>'
        xml += "</body>" * numCloses  # Close links
        xml += """
        </body>  <!-- Close base -->
    <site name="startPos0" pos="0 1 -1" size="0.02" rgba="0 0 1 1"/>
    <site name="goalPos0" pos="-2 0 -1" size="0.02" rgba="1 0 0 1"/>
    <site name="startPos1" pos="0 1 -1" size="0.02" rgba="0 0 1 1"/>
    <site name="goalPos1" pos="-2 0 -1" size="0.02" rgba="1 0 0 1"/>    
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

numLinks = 2
lengths = [0.56551564, 0.3223784]
jointTypes = [1, 0]
xml = generateXML(numLinks, lengths, jointTypes)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

actuatorIds = [model.actuator(f"motor{i}").id for i in range(numLinks)]
jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
obstacleId = model.geom('obstacle').id

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
        bounds.setLow(0, -np.pi/2)
        bounds.setHigh(0, np.pi/2)
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

startPoses = [np.array([0.41, 0.21, 0.3], dtype=np.float32), np.array([0.51, 0.31, 0.8], dtype=np.float32)] 
goalPoses = [np.array([0.4, 0.2, 0.8], dtype=np.float32), np.array([0.50, 0.30, 0.3], dtype=np.float32)]

pathStatesArr = []
for startPos, goalPos in zip(startPoses, goalPoses):
    startQpos, jStart = ik_dls(model, startPos)

    # print("Goal IK")
    goalQpos, jGoal = ik_dls(model, goalPos, initialQpos=startQpos)
    # startQpos = np.array([0.2, -0.8, -0.3, 0.9])
    # goalQpos = np.array([-0.4, 0.7, 0.5, -1.0])

    i = 0
    for id in jointIds:
        data.qpos[id] = goalQpos[i]
        i+=1

    mujoco.mj_forward(model, data)
    goalError = np.linalg.norm(data.site('endEffector').xpos - goalPos)

    i = 0
    for id in jointIds:
        data.qpos[id] = startQpos[i]
        i+=1

    mujoco.mj_forward(model, data)
    startError = np.linalg.norm(data.site('endEffector').xpos - startPos)

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

    planner = og.RRTConnect(si)
    simpleSetup.setPlanner(planner)
    # print("Planner")
    simpleSetup.solve(0.8)
    planner.clear()

    foundSolution = simpleSetup.haveSolutionPath()

    if foundSolution:
        simpleSetup.simplifySolution()
        path = simpleSetup.getSolutionPath()
        length = path.length()
        path.interpolate(100)

        pathStates = []
        for i in range(path.getStateCount()):
            stateCopy = space.allocState()
            space.copyState(stateCopy, path.getState(i))
            pathStates.append(stateCopy)

        pathStatesArr.append(pathStates)

    else:
        pathStatesArr.append([])

index = 0

for i in range(len(startPoses)):
    model.site(f'startPos{i}').pos = startPoses[i]
    model.site(f'goalPos{i}').pos = goalPoses[i]

viewer = mujoco.viewer.launch_passive(model, data)

viewer.cam.lookat[:] = model.stat.center
viewer.cam.distance = model.stat.extent * 2
viewer.cam.elevation = -35
viewer.cam.azimuth = 145

mujoco.mj_forward(model,data)
viewer.sync()

input("Press enter to continue...")

for pathStates in pathStatesArr:
    index = 0
    while viewer.is_running() and index < len(pathStates):
        for i, jid in enumerate(jointIds):
            if not isSO2[i]:
                print(pathStates[index][i][0])
            data.qpos[jid] = pathStates[index][i].value if isSO2[i] else pathStates[index][i][0]

        mujoco.mj_forward(model, data)
        viewer.sync()

        time.sleep(0.05)
        index += 1

print("Sim Complete")

while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)
    