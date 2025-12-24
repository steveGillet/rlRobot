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
    max_iters: int = 500,  # Increased for harder convergence
    tol: float = 1e-3,
    lambda_: float = 1e-2,
    max_step: float = 0.3,
    penalty_weight: float = 10.0,
    safety_margin: float = 0.01,
    fd_eps: float = 1e-4,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    site_id = model.site("endEffector").id
    nq = model.nq
    nv = model.nv

    obstacle_ids = set([model.geom(name).id for name in ["mountWall", "shelfWall", "shelf", "floor"]])

    ik_data = mujoco.MjData(model)

    if initialQpos is None:
        q = np.zeros(nq)
    else:
        q = initialQpos.copy()
        if not np.all(np.isfinite(q)):
            q = np.zeros(nq)

    q_min = np.full(nq, -np.inf)
    q_max = np.full(nq, np.inf)
    for j in range(model.njnt):
        adr = model.jnt_qposadr[j]
        if model.jnt_limited[j]:
            q_min[adr], q_max[adr] = model.jnt_range[j]

    q = np.clip(q, q_min, q_max)

    q_best = q.copy()
    err_best = np.inf

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for _ in range(max_iters):
        if not np.all(np.isfinite(q)):
            return None, None

        ik_data.qpos[:] = q
        mujoco.mj_forward(model, ik_data)
        current_pos = ik_data.site(site_id).xpos.copy()

        pos_err = target_pos - current_pos
        pos_err_norm = np.linalg.norm(pos_err)

        penalty = 0.0
        for j in range(ik_data.ncon):
            contact = ik_data.contact[j]
            if contact.geom1 in obstacle_ids or contact.geom2 in obstacle_ids:
                dist = contact.dist
                if dist < safety_margin:
                    penetration = safety_margin - dist
                    penalty += penetration ** 2

        total_err = pos_err_norm + penalty_weight * penalty
        if total_err < err_best:
            err_best = total_err
            q_best = q.copy()

        if pos_err_norm < tol and penalty == 0.0:
            break

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = jacp[:, :nv]
        A = J @ J.T + (lambda_ ** 2) * np.eye(3)
        try:
            v = np.linalg.solve(A, pos_err)
        except np.linalg.LinAlgError:
            break
        dq_pos = J.T @ v

        dq_penalty = np.zeros(nv)
        if penalty > 0.0:
            base_penalty = penalty
            for i in range(nv):
                q_pert = q.copy()
                q_pert[i] += fd_eps
                q_pert = np.clip(q_pert, q_min, q_max)
                ik_data.qpos[:] = q_pert
                mujoco.mj_forward(model, ik_data)
                pert_penalty = 0.0
                for j in range(ik_data.ncon):
                    contact = ik_data.contact[j]
                    if contact.geom1 in obstacle_ids or contact.geom2 in obstacle_ids:
                        dist = contact.dist
                        if dist < safety_margin:
                            pert_penalty += (safety_margin - dist) ** 2
                grad_i = (pert_penalty - base_penalty) / fd_eps
                dq_penalty[i] = -grad_i

        dq = dq_pos + penalty_weight * dq_penalty

        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq *= max_step / (step_norm + 1e-8)

        q += dq
        q = np.clip(q, q_min, q_max)

    if not np.all(np.isfinite(q_best)):
        return None, None

    # Recompute errors for q_best (fix from original)
    ik_data.qpos[:] = q_best
    mujoco.mj_forward(model, ik_data)
    current_pos = ik_data.site(site_id).xpos.copy()
    pos_err_norm = np.linalg.norm(target_pos - current_pos)
    penalty = 0.0
    for j in range(ik_data.ncon):
        contact = ik_data.contact[j]
        if contact.geom1 in obstacle_ids or contact.geom2 in obstacle_ids:
            dist = contact.dist
            if dist < safety_margin:
                penetration = safety_margin - dist
                penalty += penetration ** 2
    print(f"Best error: pos={pos_err_norm:.4f}, penalty={penalty:.4f}")

    try:
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        jBest = jacp[:, :nv]
        if not np.all(np.isfinite(jBest)):
            jBest = np.zeros((3, nv))
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
        <light diffuse=".5 .5 .5" pos="3 1 2" dir="0 0 -1" cutoff="180"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 0.5 0 1"/>
        <!-- <geom name="containerBack" type="box" pos="-2.0 0.6 1.0" size="0.01 1.0 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="containerLeft" type="box" pos="0 -0.4 1.0" size="2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="containerRight" type="box" pos="0 1.6 1.0" size=" 2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="containerTop" type="box" pos="0 0.6 2.0" size="2.0 1.0 0.01" rgba="0.5 0.5 0.5 1"/> -->
        <geom name="mountWall" type="box" pos="0 -0.4 1.0" size="1.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="shelfWall" type="box" pos="0 1.6 1.0" size=" 2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="shelf" type="box" pos="0.0 1.35 1.0" size="2.0 0.25 0.01" rgba="0.5 0.5 0.5 1"/>
        <body name="base" pos="0 -0.4 1.0" euler="-1.57 0 0">
        <!-- <geom name="obstacle" type="box" pos="0.45 0.25 0.55" size="0.3 0.1 0.025" rgba="1 0.5 0 1" /> -->
        <!-- <body name="base" pos="0 0 0"> -->
            <geom name="baseBox" type="box" size="0.1 0.1 0.05"/>
        """
        currentPos = "0 0 0.05"
        numCloses = 0
        for i in range(numJoints):
            if jointTypes[i] == 0:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="1 0 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses +=1
            elif jointTypes[i] == 1:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 2:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 0 1" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            else:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <geom name="baseCapsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]/2}"/>
                    <body name="slideChild{i}"> 
                        <joint name="joint{i}" type="slide" axis="0 0 1" range="0 {lengths[i]}" damping="1.0"/>
                        <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]/2}"/>
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

# numLinks = 7
# sizeMultiplier = 2
# lengths = sizeMultiplier * np.array([0.333, 0.316, 0.0825, 0.0825, 0.384, 0.088, 0.01])
# jointTypes = np.array([2, 1, 2, 0, 2, 1, 2])
numLinks = 7
lengths = np.array([0.05, 0.05, 0.05, 0.05, 0.46852955, 0.05, 1.1999999])
jointTypes = np.array([3, 0, 1, 2, 3, 3, 0])
xml = generateXML(numLinks, lengths, jointTypes)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

actuatorIds = [model.actuator(f"motor{i}").id for i in range(numLinks)]
jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
obstacleNames = ["mountWall", "shelfWall", "shelf", "floor"]
obstacleIds = set([model.geom(name).id for name in obstacleNames])

space = ob.CompoundStateSpace()

isSO2 = []

for link in range(numLinks):
    if jointTypes[link] == 2:
        space.addSubspace(ob.SO2StateSpace(), 1.0 / 6.28)
        isSO2.append(True)
    elif jointTypes[link] == 3:
        subspace = ob.RealVectorStateSpace(1)
        space.addSubspace(subspace, 1.0 / float(lengths[link]))
        bounds = ob.RealVectorBounds(1)
        bounds.setLow(0, 0)
        bounds.setHigh(0, lengths[link])
        subspace.setBounds(bounds)
        isSO2.append(False)
    else:
        subspace = ob.RealVectorStateSpace(1)
        space.addSubspace(subspace, 1.0 / 4.71)
        bounds = ob.RealVectorBounds(1)
        bounds.setLow(0, -2.355)
        bounds.setHigh(0, 2.355)
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
        if contact.geom1 in obstacleIds or contact.geom2 in obstacleIds:
            return False
    return True

validityChecker = ob.StateValidityCheckerFn(isStateValid)
si = ob.SpaceInformation(space)
si.setStateValidityChecker(validityChecker)
simpleSetup = og.SimpleSetup(si)

# startPoses = [np.array([-1.8, 0.3, 0.3], dtype=np.float32), np.array([-1.8, 0.8, 0.4], dtype=np.float32)] 
# goalPoses = [np.array([1.9, 0.9, 0.4], dtype=np.float32), np.array([1.8, 0.31, 0.2], dtype=np.float32)]
startPoses = [np.array([-0.9, 1.35, 1.1], dtype=np.float32), np.array([-1.5, -0.4, 0.1], dtype=np.float32)] 
goalPoses = [np.array([1.5, -0.4, 0.2], dtype=np.float32), np.array([1.75, 1.36, 1.11], dtype=np.float32)]
# startPoses = [np.array([-1.0, 0.6, 0.6], dtype=np.float32)]
# goalPoses = [np.array([2.0, 0.4, 0.2], dtype=np.float32)]

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

    for i in range(len(startQpos)):
        if isSO2[i]:
            startQpos[i] = np.arctan2(np.sin(startQpos[i]), np.cos(startQpos[i]))
            goalQpos[i] = np.arctan2(np.sin(goalQpos[i]), np.cos(goalQpos[i]))

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
    planner.setRange(0.5)
    simpleSetup.setPlanner(planner)
    # print("Planner")
    simpleSetup.solve(10.0)
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

# steps = 100
# sleepTime = 0.02

# for joint in range(numLinks):
#     print(f"Rotating joint {joint} (type {jointTypes[joint]})")

#     # data.qpos[:] = 0.0
#     mujoco.mj_forward(model, data)
#     viewer.sync()
#     time.sleep(0.5)

#     if jointTypes[joint] == 2:
#         minAng = -2*np.pi
#         maxAng = 2*np.pi
#     elif jointTypes[joint] == 3:
#         minAng = 0
#         maxAng = lengths[joint]
#     else:
#         minAng = model.jnt_range[jointIds[joint]][0]
#         maxAng = model.jnt_range[jointIds[joint]][1]

#     for s in range(steps):
#         ang = minAng + (maxAng - minAng) * s / (steps - 1)
#         data.qpos[jointIds[joint]] = ang
#         mujoco.mj_forward(model,data)
#         viewer.sync()
#         time.sleep(sleepTime)

#     time.sleep(0.5)

#     for s in range(steps - 1, -1, -1):
#         ang = minAng + (maxAng - minAng) * s / (steps - 1)
#         data.qpos[jointIds[joint]] = ang
#         mujoco.mj_forward(model, data)
#         viewer.sync()
#         time.sleep(sleepTime)

#     time.sleep(1.0)


while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)
    
