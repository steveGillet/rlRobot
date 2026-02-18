import mujoco
import mujoco.viewer
import time
import numpy as np

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
    targetPose: np.ndarray,
    initialQpos: np.ndarray | None = None,
    maxIter: int = 200,
    tol: float = 0.01,
    lambda_: float = 0.01,
    alpha: float = 0.75,
    rotWeight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    endEffectorId = model.site("endEffector").id

    if initialQpos is not None:
        data.qpos[:] = initialQpos.copy()
    else:
        data.qpos[:] = np.zeros(model.nq)

    mujoco.mj_forward(model, data)
    posError = targetPose[:3] - data.site(endEffectorId).xpos.copy()
    currentQuat = np.zeros(4)
    mujoco.mju_mat2Quat(currentQuat, data.site(endEffectorId).xmat.flatten())
    targetQuat = targetPose[3:7]
    rotError = np.zeros(3)
    mujoco.mju_subQuat(rotError, targetQuat, currentQuat)
    deltaX = np.concatenate([posError, rotError*rotWeight])

    i = 0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    while i < maxIter and np.linalg.norm(deltaX) > tol:
        mujoco.mj_jacSite(model, data, jacp, jacr, endEffectorId)
        J = np.vstack([jacp, jacr])

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
            data.qpos[:] = normalizeQ(model, data.qpos)
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

        posError = targetPose[:3] - data.site(endEffectorId).xpos.copy()
        currentQuat = np.zeros(4)
        mujoco.mju_mat2Quat(currentQuat, data.site(endEffectorId).xmat.flatten())
        targetQuat = targetPose[3:7]
        rotError = np.zeros(3)
        mujoco.mju_subQuat(rotError, targetQuat, currentQuat)
        deltaX = np.concatenate([posError, rotError*rotWeight])
        i += 1

    mujoco.mj_jacSite(model, data, jacp, jacr, endEffectorId)
    J = np.vstack([jacp, jacr])
    return data.qpos.copy(), J.copy()

def robustDLSik(
    model,
    obstacleIds,
    targetPose: np.ndarray,
    initialQpos: np.ndarray | None = None,
    maxIter: int = 200,
    tol: float = 0.01,
    lambda_: float = 0.01,
    alpha: float = 0.75,
    numTries: int = 5,
    rotWeight: float = 0.1,
):
    bestQpos, bestJ, bestError = None, None, np.inf

    if initialQpos is not None:
        initQ = initialQpos
    else:
        initQ = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        for i in range(len(initQ)):
            if not model.jnt_limited[i]:
                initQ[i] = np.random.uniform(-np.pi, np.pi)

    for _ in range(numTries):
        qPos, J = dlsIK(
            model,
            obstacleIds,
            targetPose,
            initialQpos=initQ,
            maxIter=maxIter,
            tol=tol,
            lambda_=lambda_,
            alpha=alpha,
            rotWeight=rotWeight,
        )

        data = mujoco.MjData(model)
        data.qpos[:] = qPos
        mujoco.mj_forward(model, data)
        posError = targetPose[:3] - data.site("endEffector").xpos.copy()
        currentQuat = np.zeros(4)
        mujoco.mju_mat2Quat(currentQuat, data.site("endEffector").xmat.flatten())
        targetQuat = targetPose[3:7]
        rotError = np.zeros(3)
        mujoco.mju_subQuat(rotError, targetQuat, currentQuat)
        totalError = np.linalg.norm(posError) + np.linalg.norm(rotError) * rotWeight

        if totalError < bestError:
            bestError = totalError
            bestQpos = qPos
            bestJ = J

        initQ = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        for i in range(len(initQ)):
            if not model.jnt_limited[i]:
                initQ[i] = np.random.uniform(-np.pi, np.pi)
    
    return bestQpos, bestJ, bestError

# ───────────────
# XML generation
# ───────────────
import math

def generateXML(numJoints, lengths, jointTypes):
    try:
        # --- Procedural High-Res Cone Mesh ---
        sides = 16
        vertices = ["0 0 1"]
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            vertices.append(f"{0.05 * math.cos(angle):.4f} {0.05 * math.sin(angle):.4f} 0")
        vertices.append("0 0 0")
        vertex_str = "  ".join(vertices)
        
        faces = []
        for i in range(sides):
            v1, v2 = i + 1, (i + 1) % sides + 1
            faces.append(f"0 {v1} {v2}")
            faces.append(f"{sides + 1} {v2} {v1}")
        face_str = "  ".join(faces)

        xml = f"""
<mujoco>
    <compiler angle="radian" />
    <option gravity="0 0 -9.81" />

    <visual>
        <quality shadowsize="8192" numslices="32" numquads="4"/>
        <map shadowscale="0.4" shadowclip="5.0"/>
        <global offwidth="3840" offheight="2160" />
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.95 0.95 0.95" rgb2="0.9 0.9 0.9" width="512" height="512" />
        <material name="floorMat" texture="grid" texrepeat="15 15" specular="0.1" shininess="0.1" reflectance="0.01" />
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.85 0.85 0.9" width="512" height="512"/>

        <material name="robotMat" rgba="0.3 0.35 0.4 1" specular="0.7" shininess="0.8" reflectance="0.2" />
        <material name="shelfMat" rgba="0.4 0.25 0.15 1" specular="0.2" shininess="0.1" />
        <material name="startMat" rgba="0 0.6 1 0.6" emission="0.4" />
        <material name="goalMat" rgba="1 0.2 0.2 0.6" emission="0.4" />
        
        <mesh name="coneMesh" vertex="{vertex_str}" face="{face_str}" scale="0.4 0.4 0.15" /> 
    </asset>

    <worldbody>
        <light name="shadow_caster" directional="true" pos="0 0 5" dir="-1 -1 -2" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1" castshadow="true" />
        
        <light name="ambient_pool" pos="0 0 4" dir="0 0 -1" diffuse="0.5 0.5 0.5" specular="0.3 0.3 0.3" castshadow="false" cutoff="60" />
        
        <light name="rim" pos="-3 -3 3" dir="1 1 -1" diffuse="0.2 0.2 0.2" castshadow="false" />

        <camera name="paper_cam" pos="2.5 -2.0 1.5" xyaxes="0.7 0.7 0.0 -0.3 0.3 0.9" />

        <geom name="floor" type="plane" size="5 5 0.1" material="floorMat" />
        <geom name="floatingShelf" type="box" pos="1 -0.5 0.5" size="0.25 0.5 0.01" material="shelfMat" />

        <body name="base" pos="0 0 0.06">
            <geom name="baseBox" type="box" size="0.12 0.12 0.06" material="robotMat" />
        """

        currentPos = "0 0 0.06"
        numCloses = 0
        for i in range(numJoints):
            axis = "1 0 0" if jointTypes[i] == 0 else ("0 1 0" if jointTypes[i] == 1 else "0 0 1")
            jtype = "slide" if jointTypes[i] == 3 else "hinge"
            
            xml += f"""
            <body name="link{i}" pos="{currentPos}">
                <joint name="joint{i}" type="{jtype}" axis="{axis}" damping="1.0" />
                <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
            """
            currentPos = f"0 0 {lengths[i]}"
            numCloses += 1

        xml += f'<site name="endEffector" pos="{currentPos}" size="0.015" rgba="0.2 1 0.2 1" />'
        xml += "</body>" * numCloses
        
        marker_cfg = 'type="mesh" mesh="coneMesh" contype="0" conaffinity="0" group="1"'
        xml += f"""
        </body>
        <body name="start0" pos="1.1 -0.31 0.61" quat="0.707 0 0.707 0"><geom {marker_cfg} material="startMat"/></body>
        <body name="start1" pos="0.95 -0.29 0.41" quat="0.707 0 0.707 0"><geom {marker_cfg} material="startMat"/></body>
        <body name="goal0" pos="0.9 -0.71 0.41" quat="0.707 0 0.707 0"><geom {marker_cfg} material="goalMat"/></body>
        <body name="goal1" pos="1.05 -0.69 0.61" quat="0.707 0 0.707 0"><geom {marker_cfg} material="goalMat"/></body>
    </worldbody>
    <actuator>
        """
        for i in range(numJoints):
            xml += f'<motor name="motor{i}" joint="joint{i}" ctrlrange="-10 10"/>'
        xml += "</actuator></mujoco>"
        return xml
    except Exception as e:
        print(f"Mujoco XML Generation Error: {e}")
        raise

# ─────────────
# Main script
# ─────────────
# numLinks = 2
# lengths = np.array([0.77285725, 1.1999999])
# jointTypes = np.array([1, 0])
# # PANDA
# numLinks = 7
# sizeMultiplier = 2
# lengths = sizeMultiplier * np.array([0.333, 0.316, 0.0825, 0.0825, 0.384, 0.088, 0.01])
# jointTypes = np.array([2, 1, 2, 0, 2, 0, 2])
# FANUC
numLinks = 6
sizeMultiplier = 2
lengths = sizeMultiplier * np.array([0.165, 0.330, 0.08, 0.285, 0.05, 0.05])
jointTypes = np.array([2, 0, 0, 2, 0, 2])

xml = generateXML(numLinks, lengths, jointTypes)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
# obstacleNames = ["mountWall", "shelfWall", "shelf", "floor"]
obstacleNames = ["floatingShelf", "floor"]
obstacleIds = set(model.geom(name).id for name in obstacleNames)

# # wall mount task
# startPoses = [
#     np.array([-1.4, -0.4, 0.1, -0.717, -0.717, 0, 0], dtype=np.float32),
#     np.array([-0.9, 1.35, 1.1, 1, 0, 0, 0], dtype=np.float32)  
# ]
# goalPoses = [
#     np.array([1.75, 1.36, 1.11, 1, 0, 0, 0], dtype=np.float32),
#     np.array([1.5, -0.4, 0.2, -0.717, -0.717, 0, 0], dtype=np.float32)
# ]

# startPoses = [np.array([-1.8, 0.3, 0.3, 0.7071, 0, -0.7071, 0], dtype=np.float32), np.array([-1.8, 0.8, 0.4, 0.7071, 0, -0.7071, 0], dtype=np.float32)] 
# goalPoses = [np.array([1.9, 0.9, 0.4, 0.7071, 0, 0.7071, 0], dtype=np.float32), np.array([1.8, 0.31, 0.2, 0.7071, 0, 0.7071, 0], dtype=np.float32)]

# Simple Shelf Task
startPoses = [
    np.array([1.1, -0.31, 0.6, 0.717, 0, 0.717, 0], dtype=np.float32),
    np.array([0.95, -0.29, 0.4, 0.717, 0, 0.717, 0], dtype=np.float32),
]
goalPoses = [
    np.array([0.9, -0.71, 0.4, 0.717, 0, 0.717, 0], dtype=np.float32),
    np.array([1.05, -0.69, 0.6, 0.717, 0, 0.717, 0], dtype=np.float32),
]

pathLists = []  # List of paths (each is a list of qpos arrays)

startQposes = []
goalQposes = []

for startPose, goalPose in zip(startPoses, goalPoses):
    print(f"\nPlanning from {startPose} to {goalPose}...")

    # Solve IK for start and goal
    startQpos, _, startError = robustDLSik(model, obstacleIds, startPose)
    goalQpos, _, goalError = robustDLSik(model, obstacleIds, goalPose, initialQpos=startQpos)
    
    # Start Error
    data.qpos[:] = startQpos
    mujoco.mj_forward(model, data)
    actualStartPos = data.site("endEffector").xpos.copy()
    print(f"Target start: {startPose}, Actual: {actualStartPos}, Error: {startError}")

    # Same for goalQpos after its IK
    data.qpos[:] = goalQpos
    mujoco.mj_forward(model, data)
    actualGoalPos = data.site("endEffector").xpos.copy()
    print(f"Target goal: {goalPose}, Actual: {actualGoalPos}, Error: {goalError}")

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
# for i in range(len(startPoses)):
#     model.site(f'startPos{i}').pos = startPoses[i][:3]
#     model.site(f'goalPos{i}').pos = goalPoses[i][:3]

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