import mujoco
import mujoco.viewer
import time
import numpy as np
import math
import ompl.base as ob
import ompl.geometric as og
from ompl import util as ou
ou.setLogLevel(ou.LOG_NONE)

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

        if np.random.rand() < 0.1:
            qRand = treeB[-1].copy() 
        else:
            qRand = np.random.uniform(low, high)

        # Find nearest neighbor
        nearestNeighbor = findNearest(model, treeA, qRand)
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
        nearestNeighbor = findNearest(model, treeB, qRand)
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
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    
    for j in range(data.ncon):
        contact = data.contact[j]
        if contact.dist >= 0:          # only real penetrations
            continue
        
        g1, g2 = contact.geom1, contact.geom2
        
        if g1 in obstacleIds or g2 in obstacleIds:
            return True
        
        # Self-collision between non-adjacent robot bodies
        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]
        if abs(b1 - b2) > 1:
            return True
            
    return False

def takeStep(model, qNear, qRand, stepSize):
    diff = np.array(qRand) - np.array(qNear)
    wrappedDiff = normalizeQ(model, diff)
    dist = np.linalg.norm(wrappedDiff)

    if dist <= stepSize:
        qNew = qRand.copy()
    else:
        dir = wrappedDiff / dist
        qNew = qNear + dir * stepSize

    for i in range(model.nq):
        if model.jnt_limited[i]:
            qNew[i] = np.clip(qNew[i], model.jnt_range[i, 0], model.jnt_range[i, 1])
            
    return qNew

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

def findNearest(model, tree, qRand):
    # Convert the entire tree list to a 2D NumPy array
    treeArray = np.array(tree)
    
    # Broadcast subtraction: calculates difference for every node at once
    diff = qRand - treeArray
    
    # Create a boolean mask for joints that don't have limits
    unlimitedMask = ~np.array(model.jnt_limited[:len(qRand)], dtype=bool)
    
    # Apply angle wrapping strictly to the unlimited joints across all nodes
    diff[:, unlimitedMask] = (diff[:, unlimitedMask] + np.pi) % (2 * np.pi) - np.pi
    
    # Compute the Euclidean distance across the rows (axis=1)
    distances = np.linalg.norm(diff, axis=1)
    
    # Return the index of the minimum distance
    return np.argmin(distances)

def omplRRTConnect(model, data, qStart, qGoal, obstacleIds, totalTime=2.0, numInterpolatedStates=100, maxGoalDist=0.1):
    nq = model.nq
    space = ob.CompoundStateSpace()
    isSO2 = []

    for i in range(nq):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE:
            # Prismatic (slide) joint
            subspace = ob.RealVectorStateSpace(1)
            bounds = ob.RealVectorBounds(1)
            bounds.setLow(0, model.jnt_range[i, 0])
            bounds.setHigh(0, model.jnt_range[i, 1])
            subspace.setBounds(bounds)
            space.addSubspace(subspace, 1.0)
            isSO2.append(False)
        else:
            # Hinge joint
            if model.jnt_limited[i]:
                # Bounded hinge
                subspace = ob.RealVectorStateSpace(1)
                bounds = ob.RealVectorBounds(1)
                bounds.setLow(0, model.jnt_range[i, 0])
                bounds.setHigh(0, model.jnt_range[i, 1])
                subspace.setBounds(bounds)
                space.addSubspace(subspace, 1.0)
                isSO2.append(False)
            else:
                # Unlimited rotation → SO2 (handles wrapping automatically)
                space.addSubspace(ob.SO2StateSpace(), 1.0)
                isSO2.append(True)

    def isStateValid(state):
        qpos = np.zeros(nq)
        for i in range(nq):
            if isSO2[i]:
                qpos[i] = state[i].value
            else:
                qpos[i] = state[i][0]
        if not np.all(np.isfinite(qpos)):
            return False
        # Use your modern collision checker (distinguishes real obstacles vs. joint contacts)
        return not checkCollision(model, data, qpos, obstacleIds)

    validityChecker = ob.StateValidityCheckerFn(isStateValid)
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(validityChecker)

    # Normalize angles for SO2 joints (same safety step you used in the old code)
    qStart = qStart.copy()
    qGoal = qGoal.copy()
    for i in range(nq):
        if isSO2[i]:
            qStart[i] = np.arctan2(np.sin(qStart[i]), np.cos(qStart[i]))
            qGoal[i] = np.arctan2(np.sin(qGoal[i]), np.cos(qGoal[i]))

    # Build start / goal states (exact syntax from your original OMPL code)
    start = ob.State(space)
    goal = ob.State(space)
    for i in range(nq):
        if isSO2[i]:
            start()[i].value = float(qStart[i])
            goal()[i].value = float(qGoal[i])
        else:
            start()[i][0] = float(qStart[i])
            goal()[i][0] = float(qGoal[i])

    simpleSetup = og.SimpleSetup(si)
    simpleSetup.setStartAndGoalStates(start, goal, 0.05)

    planner = og.RRTConnect(si)
    simpleSetup.setPlanner(planner)

    solved = simpleSetup.solve(totalTime)
    if solved:
        simpleSetup.simplifySolution()
        path = simpleSetup.getSolutionPath()
        path.interpolate(numInterpolatedStates)

        # Convert OMPL path back to list of numpy qpos arrays (exactly what your reward code expects)
        qpath = []
        for k in range(path.getStateCount()):
            state = path.getState(k)
            q = np.zeros(nq)
            for i in range(nq):
                if isSO2[i]:
                    q[i] = state[i].value
                else:
                    q[i] = state[i][0]
            qpath.append(q)

        finalDist = cDist(model, qpath[-1], qGoal)   # your existing cDist function
        print(f"  Final config distance to exact IK goal: {finalDist:.4f} rad")

        if finalDist < maxGoalDist:
            print(f"  → ACCEPTED (within tolerance {maxGoalDist})")
            return True, qpath
        else:
            print(f"  → REJECTED (too far: {finalDist:.4f} > {maxGoalDist})")
            return False, []

    else:
        return False, []

# ────────────
# IK function 
# ────────────
def dlsIK(
    model,
    data,
    obstacleIds,
    targetPose: np.ndarray,
    initialQpos: np.ndarray | None = None,
    maxIter: int = 150,
    tol: float = 0.005,
    lambda_: float = 0.1,
    alpha: float = 0.5,
    rotWeight: float = 0.1,
    avoidance_gain: float = 1.0,      # ← NEW: 0.0 disables, 0.5-2.0 is good
    repulsion_range: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    endEffectorId = model.site("endEffector").id

    if initialQpos is not None:
        data.qpos[:] = initialQpos.copy()
    else:
        data.qpos[:] = np.zeros(model.nq)

    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for i in range(maxIter):
        posError = targetPose[:3] - data.site(endEffectorId).xpos.copy()
        currentQuat = np.zeros(4)
        mujoco.mju_mat2Quat(currentQuat, data.site(endEffectorId).xmat.flatten())
        targetQuat = targetPose[3:7]
        rotError = np.zeros(3)
        mujoco.mju_subQuat(rotError, targetQuat, currentQuat)
        
        deltaX = np.concatenate([posError, rotError * rotWeight])
        
        if np.linalg.norm(deltaX) < tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, endEffectorId)
        J = np.vstack([jacp, jacr])

        U, Sigma, VT = np.linalg.svd(J, compute_uv=True, full_matrices=False)
        D = np.diag(Sigma / (Sigma**2 + lambda_**2))

        deltaTheta = VT.T @ D @ U.T @ deltaX

        # ──────── NULLSPACE OBSTACLE REPULSION (NEW) ────────
        if avoidance_gain > 0.0:
            repulsion = np.zeros(model.nv)
            jacp_contact = np.zeros((3, model.nv))
            jacr_contact = np.zeros((3, model.nv))
            
            # Loop through MuJoCo's native surface-to-surface contacts
            for j in range(data.ncon):
                contact = data.contact[j]
                g1, g2 = contact.geom1, contact.geom2
                
                is_g1_obs = g1 in obstacleIds
                is_g2_obs = g2 in obstacleIds
                
                # Identify which is the robot and which is the obstacle
                if is_g1_obs and not is_g2_obs:
                    robot_geom, sign = g2, 1.0  # Normal points away from wall
                elif is_g2_obs and not is_g1_obs:
                    robot_geom, sign = g1, -1.0 # Normal points toward wall, so flip it
                else:
                    continue
                    
                dist = contact.dist
                
                # contact.dist is the exact surface-to-surface distance!
                if dist < repulsion_range:
                    safe_dist = max(dist, 0.001)
                    
                    # Your APF magnitude formula
                    magnitude = avoidance_gain / (safe_dist ** 2 + 0.001)
                    
                    # The exact 3D direction pushing away from the surface
                    direction = contact.frame[:3] * sign
                    
                    # Get Jacobian at the EXACT contact point on the robot's surface
                    robot_body = model.geom_bodyid[robot_geom]
                    mujoco.mj_jac(model, data, jacp_contact, jacr_contact, contact.pos, robot_body)
                    
                    repulsion += magnitude * (jacp_contact.T @ direction)

            # Project repulsion into the nullspace so it NEVER disturbs the EE task
            J_pinv = VT.T @ D @ U.T   
            nullspace_proj = np.eye(model.nv) - J_pinv @ J
            deltaTheta += nullspace_proj @ repulsion

        # Update pose WITHOUT checking collisions mid-step
        data.qpos[:] = data.qpos[:] + alpha * deltaTheta
        
        for j in range(model.nq):
            if model.jnt_limited[j]:
                data.qpos[j] = np.clip(data.qpos[j], model.jnt_range[j, 0], model.jnt_range[j, 1])
        data.qpos[:] = normalizeQ(model, data.qpos)
        
        mujoco.mj_forward(model, data)

    mujoco.mj_jacSite(model, data, jacp, jacr, endEffectorId)
    J = np.vstack([jacp, jacr])
    return data.qpos.copy(), J.copy()

def robustDLSik(
    model,
    data,
    obstacleIds,
    targetPose: np.ndarray,
    initialQpos: np.ndarray | None = None,
    maxIter: int = 50,
    tol: float = 0.005,
    lambda_: float = 0.1,
    alpha: float = 0.5,
    numTries: int = 25,
    rotWeight: float = 0.1,
):
    bestQpos, bestJ, bestError = None, None, np.inf

    for t in range(numTries):
        if initialQpos is not None and t == 0:
            initQ = initialQpos
        else:
            initQ = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
            for i in range(len(initQ)):
                if not model.jnt_limited[i]:
                    initQ[i] = np.random.uniform(-np.pi, np.pi)

        qPos, J = dlsIK(
            model, data, obstacleIds, targetPose,
            initialQpos=initQ, maxIter=maxIter, tol=tol,
            lambda_=lambda_, alpha=alpha, rotWeight=rotWeight,
        )

        # Evaluate the solved pose
        data.qpos[:] = qPos
        mujoco.mj_forward(model, data)
        
        # Check if the final solved pose is in collision
        collision = checkCollision(model, data, qPos, obstacleIds)

        posError = targetPose[:3] - data.site("endEffector").xpos.copy()
        currentQuat = np.zeros(4)
        mujoco.mju_mat2Quat(currentQuat, data.site("endEffector").xmat.flatten())
        targetQuat = targetPose[3:7]
        rotError = np.zeros(3)
        mujoco.mju_subQuat(rotError, targetQuat, currentQuat)
        
        totalError = np.linalg.norm(posError) + np.linalg.norm(rotError) * rotWeight
        
        # Heavily penalize colliding solutions so they are rejected
        if collision:
            totalError += 10.0  

        if totalError < bestError:
            bestError = totalError
            bestQpos = qPos
            bestJ = J
            
        # Early exit if we hit a great, collision-free solution
        if bestError < tol:
            break
    
    return bestQpos, bestJ, bestError

# ───────────────
# XML generation
# ───────────────

# A central registry of all tasks defined in the paper
TASK_REGISTRY = {
"container": {
        "basePos": "0 0 0.06",
        "baseEuler": "0 0 0",
        "lightPos": "0 0 3",
        "lightDir": "-1 -1 -2",
        "obstacles": [
            {"name": "backWall", "pos": "-0.6 0.2 0.4", "size": "0.01 0.4 0.4", "rgba": "0.13 0.35 0.13 1"},
            {"name": "leftWall", "pos": "0 -0.2 0.4", "size": "0.6 0.01 0.4", "rgba": "0.13 0.35 0.13 1"},
            {"name": "rightWall", "pos": "0 0.6 0.4", "size": "0.6 0.01 0.4", "rgba": "0.13 0.35 0.13 1"},
            {"name": "ceiling", "pos": "0 0.2 0.8", "size": "0.6 0.4 0.01", "rgba": "0.13 0.35 0.13 1"},
        ],
        "starts": [
            [-0.45, 0.0, 0.2, 0.7071, 0, -0.7071, 0],
            [-0.45, 0.4, 0.3, 0.7071, 0, -0.7071, 0]
        ],
        "goals": [
            [0.45, 0.4, 0.3, 0.7071, 0, 0.7071, 0],
            [0.45, 0.0, 0.1, 0.7071, 0, 0.7071, 0]
        ]
    },
    "wallMount": {
        "basePos": "0 -0.2 0.6",
        "baseEuler": "-1.5708 0 0",
        "lightPos": "1.0 -1.0 2.0",   
        "lightDir": "-1 1 -1",       
        "obstacles": [
            {"name": "mountWall", "pos": "0 -0.2 0.6", "size": "0.3 0.01 0.6", "rgba": "0.82 0.70 0.54 1"},
            {"name": "shelfWall", "pos": "0 0.6 0.6", "size": "0.6 0.01 0.6", "rgba": "0.82 0.70 0.54 1"},  
            {"name": "shelf", "pos": "0.0 0.5 0.6", "size": "0.6 0.1 0.01", "rgba": "0.4 0.25 0.15 1"},
        ],
        "starts": [
            [-0.4, -0.2, 0.2, 0, 1, 0, 0],            
            [-0.2, 0.4, 0.7, 0.7071, -0.7071, 0, 0]    
        ],
        "goals": [
            [0.5, 0.4, 0.7, 0.7071, -0.7071, 0, 0],   
            [0.4, -0.2, 0.2, 0, 1, 0, 0]              
        ]
    },
    "shelf": {
        "basePos": "0 0 0.06",
        "baseEuler": "0 0 0",
        "obstacles": [
            {"name": "shelf", "pos": "0.4 -0.2 0.3", "size": "0.1 0.2 0.01"},
        ],
        "starts": [
            [0.45, -0.1, 0.4, 0.7071, 0, 0.7071, 0],
            [0.35, -0.1, 0.2, 0.7071, 0, 0.7071, 0]
        ],
        "goals": [
            [0.35, -0.3, 0.2, 0.7071, 0, 0.7071, 0],
            [0.45, -0.3, 0.4, 0.7071, 0, 0.7071, 0]
        ]
    },
    "outreach": {
        "basePos": "0 0 0.06",
        "baseEuler": "0 0 0",
        "obstacles": [], 
        "starts": [
            [0.2, 0.2, 0.12, 0.7071, 0, 0.7071, 0]
        ],
        "goals": [
            [0.65, 0.2, 0.12, 0.7071, 0, 0.7071, 0]
        ]
    },
    "sideToSide": {
        "basePos": "0 0 0.06",
        "baseEuler": "0 0 0",
        "obstacles": [], 
        "starts": [
            [0, -0.4, 0.3, 0.7071, 0.7071, 0, 0],
            [0, 0.4, 0.4, 0.7071, -0.7071, 0, 0]
        ],
        "goals": [
            [0, 0.4, 0.3, 0.7071, -0.7071, 0, 0],
            [0, -0.4, 0.4, 0.7071, 0.7071, 0, 0]
        ]
    }
}

def generateXML(numJoints, lengths, jointTypes, taskConfig, numGhosts=0, activePathIndex=None, solidStart=False):
    try:
        # --- Procedural High-Res Cone Mesh ---
        sides = 16
        vertices = ["0 0 1"]
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            vertices.append(f"{0.05 * math.cos(angle):.4f} {0.05 * math.sin(angle):.4f} 0")
        vertices.append("0 0 0")
        vertexStr = "  ".join(vertices)
        
        faces = []
        for i in range(sides):
            v1, v2 = i + 1, (i + 1) % sides + 1
            faces.append(f"0 {v1} {v2}")
            faces.append(f"{sides + 1} {v2} {v1}")
        faceStr = "  ".join(faces)

        # Base XML setup
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
        <material name="ghostMat" rgba="0.3 0.35 0.4 0.15" specular="0.2" reflectance="0.0" /> 
        <material name="obstacleMat" rgba="0.4 0.25 0.15 1" specular="0.2" shininess="0.1" />
        <material name="startMat" rgba="0 0.6 1 0.6" emission="0.4" />
        <material name="goalMat" rgba="1 0.2 0.2 0.6" emission="0.4" />
        
        <mesh name="coneMesh" vertex="{vertexStr}" face="{faceStr}" scale="0.4 0.4 0.15" /> 
    </asset>"""
        lightPos = taskConfig.get("lightPos", "0 0 5")
        lightDir = taskConfig.get("lightDir", "-1 -1 -2")

        xml += f"""
    <worldbody>
        <light name="shadowCaster" directional="true" pos="{lightPos}" dir="{lightDir}" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1" castshadow="true" />
        <light name="ambientPool" pos="0 0 4" dir="0 0 -1" diffuse="0.5 0.5 0.5" specular="0.3 0.3 0.3" castshadow="false" cutoff="60" />
        <geom name="floor" type="plane" size="5 5 0.1" material="floorMat" />
        """

        # Inject dynamic obstacles with custom colors
        for obs in taskConfig["obstacles"]:
            # If the registry has an RGBA string, use it. Otherwise, use the default material.
            colorStr = f'rgba="{obs["rgba"]}"' if "rgba" in obs else 'material="obstacleMat"'
            xml += f'<geom name="{obs["name"]}" type="box" pos="{obs["pos"]}" size="{obs["size"]}" {colorStr} />\n'

        # --- Helper function to stamp out robots ---
        def buildArm(prefix, matName, disableCollision, addSite=False):
            colStr = 'contype="0" conaffinity="0"' if disableCollision else ''
            
            # Keep camelCase clean: if prefix is "start", make it "startBase", else "base"
            baseName = f"{prefix}Base" if prefix else "base"
            boxName = f"{prefix}BaseBox" if prefix else "baseBox"
            
            armXml = f"""
            <body name="{baseName}" pos="{taskConfig['basePos']}" euler="{taskConfig['baseEuler']}">
                <geom name="{boxName}" type="box" size="0.12 0.12 0.06" material="{matName}" {colStr}/>
            """
            currentPos = "0 0 0.06"
            # Limit later joints to ~157 degrees
            standardLimit = 7 * math.pi / 8 
            # Limit the base joint so it cannot swing backwards into the wall
            baseLimit = math.pi / 2 
            
            for i in range(numJoints):
                li = lengths[i]
                jCode = jointTypes[i]
                
                # Apply the strict 90-degree limit only to the first joint
                currentLimit = baseLimit if i == 0 else standardLimit
                
                if jCode == 0:
                    axis, jtype, limitStr = "1 0 0", "hinge", f'limited="true" range="{-currentLimit:.4f} {currentLimit:.4f}"'
                elif jCode == 1:
                    axis, jtype, limitStr = "0 1 0", "hinge", f'limited="true" range="{-currentLimit:.4f} {currentLimit:.4f}"'
                elif jCode == 2:
                    axis, jtype, limitStr = "0 0 1", "hinge", 'limited="false"'
                elif jCode == 3:
                    axis, jtype, limitStr = "0 0 1", "slide", f'limited="true" range="0 {li:.4f}"'
                
                linkName = f"{prefix}Link{i}" if prefix else f"link{i}"
                jointName = f"{prefix}Joint{i}" if prefix else f"joint{i}"
                geomName = f"{prefix}Capsule{i}" if prefix else f"capsule{i}"
                
                armXml += f"""
                <body name="{linkName}" pos="{currentPos}">
                    <joint name="{jointName}" type="{jtype}" axis="{axis}" damping="1.0" {limitStr} />
                    <geom name="{geomName}" type="capsule" size="0.025" fromto="0 0 0 0 0 {li:.4f}" material="{matName}" {colStr}/>
                """
                currentPos = f"0 0 {li:.4f}"
            
            if addSite:
                siteName = f"{prefix}EndEffector" if prefix else "endEffector"
                armXml += f'<site name="{siteName}" pos="{currentPos}" size="0.015" rgba="0.2 1 0.2 1" />'
                
            armXml += "</body>\n" * numJoints
            armXml += "</body>\n"
            return armXml

        # 1. Build the Main robot (Goal State - Opaque, Has Physics)
        xml += buildArm("", "robotMat", disableCollision=False, addSite=True)

        # 2. Build the Start robot (Start State - Opaque, No Physics)
        if solidStart:
            xml += buildArm("start", "robotMat", disableCollision=True, addSite=False)

        # 3. Build the Ghost robots (Intermediate States - Transparent, No Physics)
        for g in range(numGhosts):
            xml += buildArm(f"ghost{g}", "ghostMat", disableCollision=True, addSite=False)

        # Inject Start and Goal markers
        markerCfg = 'type="mesh" mesh="coneMesh" contype="0" conaffinity="0" group="1"'
        if activePathIndex is not None:
            startPose = taskConfig["starts"][activePathIndex]
            goalPose = taskConfig["goals"][activePathIndex]
            xml += f'<body name="startMarker{activePathIndex}" pos="{startPose[0]} {startPose[1]} {startPose[2]}" quat="{startPose[3]} {startPose[4]} {startPose[5]} {startPose[6]}"><geom {markerCfg} material="startMat"/></body>\n'
            xml += f'<body name="goalMarker{activePathIndex}" pos="{goalPose[0]} {goalPose[1]} {goalPose[2]}" quat="{goalPose[3]} {goalPose[4]} {goalPose[5]} {goalPose[6]}"><geom {markerCfg} material="goalMat"/></body>\n'
        else:
            for i, startPose in enumerate(taskConfig["starts"]):
                xml += f'<body name="startMarker{i}" pos="{startPose[0]} {startPose[1]} {startPose[2]}" quat="{startPose[3]} {startPose[4]} {startPose[5]} {startPose[6]}"><geom {markerCfg} material="startMat"/></body>\n'
            for i, goalPose in enumerate(taskConfig["goals"]):
                xml += f'<body name="goalMarker{i}" pos="{goalPose[0]} {goalPose[1]} {goalPose[2]}" quat="{goalPose[3]} {goalPose[4]} {goalPose[5]} {goalPose[6]}"><geom {markerCfg} material="goalMat"/></body>\n'

        xml += "</worldbody><actuator>\n"
        for i in range(numJoints):
            xml += f'<motor name="motor{i}" joint="joint{i}" ctrlrange="-10 10"/>\n'
        xml += "</actuator></mujoco>"
        return xml
    except Exception as e:
        print(f"Mujoco XML Generation Error: {e}")
        raise

# ─────────────
# Main script
# ─────────────
numLinks = 2
lengths = np.array([0.3593, 0.376])
jointTypes = np.array([1, 0])
# PANDA
# numLinks = 7
# sizeMultiplier = 1
# lengths = sizeMultiplier * np.array([0.333, 0.316, 0.0825, 0.0825, 0.384, 0.088, 0.01])
# jointTypes = np.array([2, 1, 2, 0, 2, 0, 2])
# # FANUC
# numLinks = 6
# sizeMultiplier = 1
# lengths = sizeMultiplier * np.array([0.165, 0.330, 0.08, 0.285, 0.05, 0.05])
# jointTypes = np.array([2, 0, 0, 2, 0, 2])

taskConfig = TASK_REGISTRY["wallMount"]
xml = generateXML(numLinks, lengths, jointTypes, taskConfig)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
obstacleNames = ["floor"]+ [obs["name"] for obs in taskConfig["obstacles"]]
obstacleIds = set(model.geom(name).id for name in obstacleNames)

startPoses = [np.array(p, dtype=np.float32) for p in taskConfig["starts"]]
goalPoses = [np.array(p, dtype=np.float32) for p in taskConfig["goals"]]

pathLists = []  # List of paths (each is a list of qpos arrays)

startQposes = []
goalQposes = []

for startPose, goalPose in zip(startPoses, goalPoses):
    print(f"\nPlanning from {startPose} to {goalPose}...")

    # Solve IK for start and goal
    startQpos, _, startError = robustDLSik(model, data, obstacleIds, startPose)
    goalQpos, _, goalError = robustDLSik(model, data, obstacleIds, goalPose, initialQpos=startQpos)
    
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

    # # Run your RRT-Connect
    # found, path = rrtConnect(
    #     model, data,
    #     startQpos, goalQpos,
    #     obstacleIds,
    #     totalTime=2.0,      # ← increase if needed
    #     stepSize=0.1,
    #     numIsteps=5,
    #     tol=0.01
    # )

    found, path = omplRRTConnect(
        model, data, startQpos, goalQpos, obstacleIds, totalTime=2.0
    )

    if found:
        print(f"Path found! Length: {len(path)} configurations")
        pathLists.append(path)
    else:
        print("No path found within time limit")
        pathLists.append([])


# # ───────────────────────────────────────────────
# # Visualization
# # ───────────────────────────────────────────────

# viewer = mujoco.viewer.launch_passive(model, data)
# viewer.cam.lookat[:] = model.stat.center
# viewer.cam.distance = model.stat.extent * 2
# viewer.cam.elevation = -35
# viewer.cam.azimuth = 145

# data.qpos[:] = startQposes[0]
# mujoco.mj_forward(model, data)
# viewer.sync()

# input("Press Enter to play paths...")

# for path in pathLists:
#     if not path:
#         continue

#     for qpos in path:
#         if not viewer.is_running():
#             break

#         for i, jid in enumerate(jointIds):
#             data.qpos[jid] = qpos[i]

#         mujoco.mj_forward(model, data)
#         viewer.sync()
#         time.sleep(0.04)

# print("Simulation complete")

# while viewer.is_running():
#     viewer.sync()
#     time.sleep(0.02)

# ───────────────────────────────────────────────
# Static Ghost Diorama Visualization
# ───────────────────────────────────────────────
NUM_GHOSTS = 4  # Intermediate faded steps
SOLID_START = False # Set this flag here so we can use it in both places

for pathIndex, path in enumerate(pathLists):
    print(f"\n--- Generating Diorama for Task {pathIndex + 1} ---")
    
    # --- NEW LOGIC: Handle failed paths ---
    if not path:
        print(f"No RRT path found for Task {pathIndex + 1}. Visualizing IK Start and Goal poses only.")
        sampledGhostStates = []
        actualGhosts = 0
        # Grab the raw IK solutions instead of relying on the RRT path
        finalQpos = goalQposes[pathIndex]
        startQpos = startQposes[pathIndex]
        
        # Force a solid start robot to be built so we can see both poses
        render_solid_start = True 
    else:
        # 1. Sample evenly spaced indices (excluding start and end)
        if len(path) > 2:
            indices = np.linspace(1, len(path)-2, NUM_GHOSTS, dtype=int)
            sampledGhostStates = [path[i] for i in indices]
        else:
            sampledGhostStates = []
            
        actualGhosts = len(sampledGhostStates)
        finalQpos = path[-1]
        startQpos = path[0]
        render_solid_start = SOLID_START

    # 2. Generate Diorama XML
    dioramaXml = generateXML(
        numLinks, 
        lengths, 
        jointTypes, 
        taskConfig, 
        numGhosts=actualGhosts, 
        activePathIndex=pathIndex,
        solidStart=render_solid_start # Pass the dynamic flag here
    )
    
    dioModel = mujoco.MjModel.from_xml_string(dioramaXml)
    dioData = mujoco.MjData(dioModel)

    # 3. Apply coordinates to the Main Robot (Goal state)
    for i in range(numLinks):
        jointId = dioModel.joint(f"joint{i}").id
        dioData.qpos[jointId] = finalQpos[i]

    # 4. Apply coordinates to the Solid Start Robot (ONLY if it exists)
    if render_solid_start:
        for i in range(numLinks):
            jointId = dioModel.joint(f"startJoint{i}").id
            dioData.qpos[jointId] = startQpos[i]

    # 5. Apply coordinates to the Ghost Robots (Intermediate states)
    for g, ghostQpos in enumerate(sampledGhostStates):
        for i in range(numLinks):
            jointId = dioModel.joint(f"ghost{g}Joint{i}").id
            dioData.qpos[jointId] = ghostQpos[i]

    # 6. Calculate forward kinematics ONCE
    mujoco.mj_forward(dioModel, dioData)

    # 7. Open Viewer
    print(f"Viewer opened for Task {pathIndex + 1}. Adjust camera, take your screenshot, then close the window to continue.")
    with mujoco.viewer.launch_passive(dioModel, dioData) as viewer:
        viewer.cam.lookat[:] = dioModel.stat.center
        viewer.cam.distance = dioModel.stat.extent * 1.5
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 135
        
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)