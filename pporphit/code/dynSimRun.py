import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.optimize import minimize

# ─────────────
# RRT-Connect
# ─────────────
def rrtConnect(model, data, qStart, qGoal, obstacleIds, totalTime=10.0, stepSize=0.1, numIsteps=100, tol=0.01):
    pathFound = False
    startTime = time.time()
    treeStart = [qStart.copy()]
    parentsTreeStart = [None]
    treeGoal = [qGoal.copy()]
    parentsTreeGoal = [None]
    path = []

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

        qRand = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
      
        # Find nearest neighbor
        nearestNeighbor = np.argmin([np.linalg.norm(np.array(q) - np.array(qRand)) for q in treeA])
        qNear = treeA[nearestNeighbor]

        qNew = takeStep(qNear, qRand, stepSize)

        # Check collision along the edge
        if not isEdgeValid(model, data, qNear, qNew, obstacleIds, numIsteps):
            continue

        # Add to tree
        treeA.append(qNew)
        parentsA.append(nearestNeighbor)

        # Try to connect other tree
        qRand = qNew.copy()
        nearestNeighbor = np.argmin([np.linalg.norm(np.array(q) - np.array(qRand)) for q in treeB])
        qNear = treeB[nearestNeighbor]

        qNew = takeStep(qNear, qRand, stepSize)

        # Check collision along the edge
        if not isEdgeValid(model, data, qNear, qNew, obstacleIds, numIsteps):
            continue

        # Add to tree
        treeB.append(qNew)
        parentsB.append(nearestNeighbor)

        treeStartTurn = not treeStartTurn

        # Check if close enough to goal
        if np.linalg.norm(np.array(qNew) - np.array(qRand)) < tol:
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
            path = interpolatePath(path)
            break

    return pathFound, path

def checkCollision(model, data, qPos, obstacleIds):
    data.qpos[:] = qPos
    mujoco.mj_forward(model, data)
    for j in range(data.ncon):
        contact = data.contact[j]
        if contact.geom1 in obstacleIds or contact.geom2 in obstacleIds:
            return True
    return False

def takeStep(qNear, qRand, stepSize):
    diff = np.array(qRand) - np.array(qNear)
    dist = np.linalg.norm(diff)
    
    if dist <= stepSize:
        return qRand  # Reach exactly if close enough
    else:
        dir = diff / dist
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
    for iStep in range(1, numIsteps + 1):
        qIntermediate = qStart + iStep / float(numIsteps) * (qEnd - qStart)
        if checkCollision(model, data, qIntermediate, obstacleIds):
            return False
    return True

def interpolatePath(path, numNodes=100):
    totalLength = 0.0
    segmentLengths = []
    for i in range(len(path) - 1):
        segmentLengths.append(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])))
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
            interpolatedPath.append(takeStep(path[i], path[i+1], step / numStepsPerSegment[i] * segmentLengths[i]))

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
            data.qpos = np.clip(data.qpos, model.jnt_range[:, 0], model.jnt_range[:, 1])
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
        <geom name="mountWall" type="box" pos="0 -0.4 1.0" size="1.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="shelfWall" type="box" pos="0 1.6 1.0" size=" 2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="shelf" type="box" pos="0.0 1.35 1.0" size="2.0 0.25 0.01" rgba="0.5 0.5 0.5 1"/>
        <body name="base" pos="0 -0.4 1.0" euler="-1.57 0 0">
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
numLinks = 7
lengths = np.array([0.05, 0.05, 0.05, 0.05, 0.46852955, 0.05, 1.1999999])
jointTypes = np.array([3, 0, 1, 2, 3, 3, 0])

xml = generateXML(numLinks, lengths, jointTypes)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
obstacleNames = ["mountWall", "shelfWall", "shelf", "floor"]
obstacleIds = set(model.geom(name).id for name in obstacleNames)

# Example start/goal positions
startPoses = [
    np.array([-0.9, 1.35, 1.1], dtype=np.float32),
    np.array([-1.4, -0.9, 0.1], dtype=np.float32)
]
goalPoses = [
    np.array([1.5, -0.4, 0.2], dtype=np.float32),
    np.array([1.75, 1.36, 1.11], dtype=np.float32)
]

pathLists = []  # List of paths (each is a list of qpos arrays)

for startPos, goalPos in zip(startPoses, goalPoses):
    print(f"\nPlanning from {startPos} to {goalPos}...")

    # Solve IK for start and goal
    startQpos, _ = dlsIK(model, obstacleIds, startPos)
    goalQpos, _ = dlsIK(model, obstacleIds, goalPos, initialQpos=startQpos)

    if startQpos is None or goalQpos is None:
        print("IK failed → skipping this pair")
        pathLists.append([])
        continue

    # Normalize angles if needed (optional - your joints are limited)
    for i in range(len(startQpos)):
        if jointTypes[i] == 2:  # revolute that might wrap
            startQpos[i] = np.arctan2(np.sin(startQpos[i]), np.cos(startQpos[i]))
            goalQpos[i] = np.arctan2(np.sin(goalQpos[i]), np.cos(goalQpos[i]))

    # Run your RRT-Connect
    found, path = rrtConnect(
        model, data,
        startQpos, goalQpos,
        obstacleIds,
        totalTime=15.0,      # ← increase if needed
        stepSize=0.08,
        greedyBias=0.25,
        numIsteps=12,
        tol=0.015
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
        time.sleep(0.04)  # slower for better visualization

print("Simulation complete")

while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)