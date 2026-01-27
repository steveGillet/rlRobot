import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.optimize import minimize

# ─────────────
# RRT-Connect
# ─────────────
def normalizeQ(model, q):
    q = q.copy()
    for i in range(len(q)):
        if not model.jnt_limited[i]:
            q[i] = (q[i] + np.pi) % (2 * np.pi) - np.pi
    return q

def cDist(model, q1, q2):
    diff = np.array(q2) - np.array(q1)
    wrappedDiff = normalizeQ(model, diff)
    return np.linalg.norm(wrappedDiff)

def rrtConnect(model, data, qStart, qGoal, obstacleIds, totalTime=10.0, stepSize=0.1, numIsteps=100, tol=0.01):
    pathFound = False
    startTime = time.time()
    treeStart = [qStart.copy()]
    parentsTreeStart = [None]
    treeGoal = [qGoal.copy()]
    parentsTreeGoal = [None]
    path = []

    # Bounds
    low = model.jnt_range[:, 0]
    high = model.jnt_range[:, 1]
    for i in range(len(low)):
        if not model.jnt_limited[i]:
            low[i] = -np.pi
            high[i] = np.pi

    treeStartTurn = True
    while not pathFound and (time.time() - startTime) < totalTime:
        if treeStartTurn:
            treeA = treeStart
            parentsA = parentsTreeStart
            treeB = treeGoal
            parentsB = parentsTreeGoal
        else:
            treeA = treeGoal
            parentsA = parentsTreeGoal
            treeB = treeStart
            parentsB = parentsTreeStart

        qRand = np.random.uniform(low, high)
        # Find nearest neighbor
        nearestNeighbor = np.argmin([cDist(model, np.array(q), np.array(qRand)) for q in treeA])
        qNear = treeA[nearestNeighbor]

        qNew = takeStep(model, qNear, qRand, stepSize)
        # Check collision along the edge
        if not isEdgeValid(model, data, qNear, qNew, obstacleIds, numIsteps):
            continue

        # Add to tree
        treeA.append(qNew)
        parentsA.append(nearestNeighbor)

        # Try to connect other tree
        terminated = False
        qRand = qNew.copy()
        nearestNeighbor = np.argmin([cDist(model, np.array(q), np.array(qRand)) for q in treeB])
        qNear = treeB[nearestNeighbor]
        while not terminated:
            qNew = takeStep(model, qNear, qRand, stepSize)

            # Check collision along the edge
            if not isEdgeValid(model, data, qNear, qNew, obstacleIds, numIsteps):
                terminated = True
                continue

            # Add to tree
            treeB.append(qNew)
            parentsB.append(nearestNeighbor)

            # Check if close enough to goal
            if cDist(model, np.array(qNew), np.array(qRand)) < tol:
                pathFound = True
                
                tempPath = []
                current = len(treeGoal) - 1
                while current is not None:
                    tempPath.append(treeGoal[current])
                    current = parentsTreeGoal[current]

                current = len(treeStart) - 1
                while current is not None:
                    path.append(treeStart[current])
                    current = parentsTreeStart[current]

                path.reverse()
                path.extend(tempPath)

                path = shortenPath(model, data, path, obstacleIds)
                path = interpolatePath(model, path)
                break

            if pathFound:
                break

            qNear = qNew
            nearestNeighbor = len(treeB) - 1

        treeStartTurn = not treeStartTurn

    return pathFound, path

def checkCollision(model, data, qPos, obstacleIds):
    data.qpos[:] = qPos
    mujoco.mj_forward(model, data)
    for j in range(data.ncon):
        contact = data.contact[j]
        if contact.geom1 in obstacleIds or contact.geom2 in obstacleIds:
            return True
    return False

def takeStep(model, qNear, qRand, stepSize):
    diff = np.array(qRand) - np.array(qNear)
    wrappedDiff = normalizeQ(model, diff)
    dist = np.linalg.norm(wrappedDiff)

    if dist <= stepSize:
        return qRand  # Reach exactly if close enough
    else:
        dir = wrappedDiff / dist
        return qNear + dir * stepSize

def shortenPath(model, data, path, obstacleIds, numIsteps=100, maxIter=20):
    if len(path) < 3:
        return path  # Too short to shorten
    
    simplified = path[:]  # Copy
    for _ in range(maxIter):
        shortened = False
        i = 0
        while i < len(simplified) - 1:
            j = len(simplified) - 1
            while j > i + 1:
                if isEdgeValid(model, data, simplified[i], simplified[j], obstacleIds, numIsteps):
                    # Shortcut: remove i+1 to j-1
                    simplified = simplified[:i+1] + simplified[j:]
                    shortened = True
                    break
                j -= 1
            if shortened:
                break  # Restart from beginning after change
            i += 1
        if not shortened:
            break  # No more improvements
    return simplified

def isEdgeValid(model, data, qStart, qEnd, obstacleIds, numIsteps):
    diff = qEnd - qStart
    wrappedDiff = normalizeQ(model, diff)
    for iStep in range(1, numIsteps + 1):
        qIntermediate = qStart + iStep / float(numIsteps) * wrappedDiff
        if checkCollision(model, data, qIntermediate, obstacleIds):
            return False
    return True

def interpolatePath(model, path, numNodes=100):
    totalLength = 0.0
    segmentLengths = []
    for i in range(len(path) - 1):
        segmentLengths.append(cDist(model, np.array(path[i]), np.array(path[i+1])))
        totalLength += segmentLengths[i]

    if totalLength == 0.0:
        return path
    
    numStepsPerSegment = []
    for i in range(len(path) - 1):
        numStepsPerSegment.append(int(np.round((segmentLengths[i] / totalLength) * (numNodes - 1))))
    
    interpolatedPath = []

    for i in range(len(path) - 1):
        interpolatedPath.append(path[i])
        for step in range(1, int(numStepsPerSegment[i])):
            interpolatedPath.append(takeStep(model, path[i], path[i+1], step / numStepsPerSegment[i] * segmentLengths[i]))

    interpolatedPath.append(path[-1])
    return interpolatedPath

# ────────────
# IK function 
# ────────────
def dlsIK(
    model,
    obstacleIds,
    targetQpos: np.ndarray,
    initialQpos: np.ndarray | None = None,
    maxIter: int = 200,
    tol: float = 0.01,
    lambda_: float = 0.01,
    alpha: float = 0.75,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    data = mujoco.MjData(model)
    endEffectorId = model.site("endEffector").id

    if initialQpos is not None:
        data.qpos[:] = initialQpos.copy()
    else:
        data.qpos[:] = np.zeros(model.nq)

    mujoco.mj_forward(model, data)
    deltaX = targetQpos - data.site(endEffectorId).xpos.copy()

    i = 0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    while i < maxIter and np.linalg.norm(deltaX) > tol:
        mujoco.mj_jacSite(model, data, jacp, jacr, endEffectorId)
        J = jacp

        U, Sigma, VT = np.linalg.svd(J, compute_uv=True, full_matrices=False)
        D = np.diag(Sigma / (Sigma**2 + lambda_**2))

        deltaTheta = VT.T @ D @ U.T @ deltaX

        collision = True
        al = alpha
        originalQpos = data.qpos.copy()

        while collision and al >= 0.01:
            collision = False

            data.qpos[:] = originalQpos + al * deltaTheta
            for j in range(model.nq):
                if model.jnt_limited[j]:
                    data.qpos[j] = np.clip(data.qpos[j], model.jnt_range[j, 0], model.jnt_range[j, 1])
            data.qpos = normalizeQ(model, data.qpos)
            mujoco.mj_forward(model, data)
            
            for j in range(data.ncon):
                contact = data.contact[j]
                if contact.geom1 in obstacleIds or contact.geom2 in obstacleIds:
                    collision = True
                    break

            if collision:
                al /= 2.0

        if collision:
            data.qpos[:] = originalQpos
            mujoco.mj_forward(model, data)
            break

        deltaX = targetQpos - data.site(endEffectorId).xpos.copy()
        i += 1

    mujoco.mj_jacSite(model, data, jacp, jacr, endEffectorId)
    J = jacp
    return data.qpos.copy(), J.copy()



# ───────────────
# XML generation
# ───────────────
def generateXML(numJoints, lengths, jointTypes):
    try:
        xml = """
<mujoco>
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="3 1 2" dir="0 0 -1" cutoff="180"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 0.5 0 1"/>
        <geom name="containerBack" type="box" pos="-2.0 0.6 1.0" size="0.01 1.0 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="containerLeft" type="box" pos="0 -0.4 1.0" size="2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="containerRight" type="box" pos="0 1.6 1.0" size=" 2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="containerTop" type="box" pos="0 0.6 2.0" size="2.0 1.0 0.01" rgba="0.5 0.5 0.5 1"/>
        <!-- <geom name="mountWall" type="box" pos="0 -0.4 1.0" size="1.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="shelfWall" type="box" pos="0 1.6 1.0" size=" 2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="shelf" type="box" pos="0.0 1.35 1.0" size="2.0 0.25 0.01" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <body name="base" pos="0 -0.4 1.0" euler="-1.57 0 0"> -->
        <body name="base" pos="0 0 0.05">
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
        xml += "</body>" * numCloses
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


# ───────────────────────────────────────────────
# Main script - now using custom RRT
# ───────────────────────────────────────────────
numLinks = 2
lengths = np.array([0.05, 1.1999999])
jointTypes = np.array([2, 0])
# # PANDA
# numLinks = 7
# sizeMultiplier = 2
# lengths = sizeMultiplier * np.array([0.333, 0.316, 0.0825, 0.0825, 0.384, 0.088, 0.01])
# jointTypes = np.array([2, 1, 2, 0, 2, 0, 2])

xml = generateXML(numLinks, lengths, jointTypes)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
# obstacleNames = ["mountWall", "shelfWall", "shelf", "floor"]
obstacleNames = ["containerTop", "containerBack", "containerLeft", "containerRight", "floor"]
obstacleIds = set(model.geom(name).id for name in obstacleNames)

# Example start/goal positions
# startPoses = [
#     np.array([-0.9, 1.35, 1.1], dtype=np.float32),
#     np.array([-1.4, -0.9, 0.1], dtype=np.float32)
# ]
# goalPoses = [
#     np.array([1.5, -0.4, 0.2], dtype=np.float32),
#     np.array([1.75, 1.36, 1.11], dtype=np.float32)
# ]

startPoses = [np.array([-1.8, 0.3, 0.3], dtype=np.float32), np.array([-1.8, 0.8, 0.4], dtype=np.float32)] 
goalPoses = [np.array([1.9, 0.9, 0.4], dtype=np.float32), np.array([1.8, 0.31, 0.2], dtype=np.float32)]

pathLists = []  # List of paths (each is a list of qpos arrays)

startQposes = []
goalQposes = []

for startPos, goalPos in zip(startPoses, goalPoses):
    print(f"\nPlanning from {startPos} to {goalPos}...")

    # Solve IK for start and goal
    startQpos, _ = dlsIK(model, obstacleIds, startPos)
    goalQpos, _ = dlsIK(model, obstacleIds, goalPos, initialQpos=startQpos)
    
    data.qpos[:] = startQpos
    mujoco.mj_forward(model, data)
    actual_start_pos = data.site("endEffector").xpos.copy()
    print(f"Target start: {startPos}, Actual: {actual_start_pos}, Error: {np.linalg.norm(actual_start_pos - startPos)}")
    print(f"Collision at start: {checkCollision(model, data, startQpos, obstacleIds)}")

    # Same for goalQpos after its IK
    data.qpos[:] = goalQpos
    mujoco.mj_forward(model, data)
    actual_goal_pos = data.site("endEffector").xpos.copy()
    print(f"Target goal: {goalPos}, Actual: {actual_goal_pos}, Error: {np.linalg.norm(actual_goal_pos - goalPos)}")
    print(f"Collision at goal: {checkCollision(model, data, goalQpos, obstacleIds)}")

    if startQpos is None or goalQpos is None:
        print("IK failed → skipping this pair")
        pathLists.append([])
        continue

    startQpos = normalizeQ(model, startQpos)
    goalQpos = normalizeQ(model, goalQpos)

    startQposes.append(startQpos)
    goalQposes.append(goalQpos)

    # Run your RRT-Connect
    found, path = rrtConnect(
        model, data,
        startQpos, goalQpos,
        obstacleIds,
        totalTime=5.0,      # ← increase if needed
        stepSize=0.1,
        numIsteps=5,
        tol=0.01
    )

    if found:
        print(f"Path found! Length: {len(path)} configurations")
        pathLists.append(path)
    else:
        print("No path found within time limit")
        pathLists.append([])


# ───────────────────────────────────────────────
# Visualization
# ───────────────────────────────────────────────
for i in range(len(startPoses)):
    model.site(f'startPos{i}').pos = startPoses[i]
    model.site(f'goalPos{i}').pos = goalPoses[i]

viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.lookat[:] = model.stat.center
viewer.cam.distance = model.stat.extent * 2
viewer.cam.elevation = -35
viewer.cam.azimuth = 145

data.qpos[:] = startQposes[0]
mujoco.mj_forward(model, data)
viewer.sync()

input("Press Enter to play paths...")

for path in pathLists:
    if not path:
        continue

    for qpos in path:
        if not viewer.is_running():
            break

        for i, jid in enumerate(jointIds):
            data.qpos[jid] = qpos[i]

        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.04)

print("Simulation complete")

while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)