import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.optimize import minimize
import gymnasium as gym
import os
import logging
import math

class robotArmEnv(gym.Env):
    def __init__(self, taskName="container", minNumLinks=2, maxNumLinks=7, minLength=0.05, maxLength=1.2, noise=0.01):
        super().__init__()
        self.minNumLinks = minNumLinks
        self.maxNumLinks = maxNumLinks
        self.minLength = minLength
        self.maxLength = maxLength
        self.noise = noise

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(1 + self.maxNumLinks * 2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.float32
        )

        if taskName not in TASK_REGISTRY:
            raise ValueError(f"Task '{taskName}' not found in TASK_REGISTRY.")
        
        self.taskConfig = TASK_REGISTRY[taskName]
        
        self.startPos = [np.array(p, dtype=np.float32) for p in self.taskConfig["starts"]]
        self.goalPos = [np.array(p, dtype=np.float32) for p in self.taskConfig["goals"]]

        self.obstacleNames = ["floor"] + [obs["name"] for obs in self.taskConfig["obstacles"]]

        self.logger = setupLogging()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def _evaluate(self, numLinks, lengths, jointTypes):
        self.logger.debug(
            f"Evaluating: numLinks={numLinks}, lengths={lengths}, jointTypes={jointTypes}"
        )

        try:
            xml = generateXML(numLinks, lengths.tolist(), jointTypes.tolist(), self.taskConfig)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
        except Exception as e:
            print(f"Mujoco XML Generation Error: {e}")
            return -50.0

        actuatorIds = [model.actuator(f"motor{i}").id for i in range(numLinks)]
        jointIds = [model.joint(f"joint{i}").id for i in range(numLinks)]
        obstacleIds = set([model.geom(name).id for name in self.obstacleNames])

        isSO2 = []

        for link in range(numLinks):
            if jointTypes[link] == 2:
                isSO2.append(True)
            else:
                isSO2.append(False)

        reward = 0
        for startPos, goalPos in zip(self.startPos, self.goalPos):
            noisyStartPos = startPos[:3] + np.random.normal(0, self.noise, size=3)
            noisyStartQuat = startPos[3:] + np.random.normal(0, self.noise, size=4)
            noisyStartQuat /= np.linalg.norm(noisyStartQuat)
            startPos = np.concatenate([noisyStartPos, noisyStartQuat])
            noisyGoalPos = goalPos[:3] + np.random.normal(0, self.noise, size=3)
            noisyGoalQuat = goalPos[3:] + np.random.normal(0, self.noise, size=4)
            noisyGoalQuat /= np.linalg.norm(noisyGoalQuat)
            goalPos = np.concatenate([noisyGoalPos, noisyGoalQuat])
            self.logger.debug(f"Pre-IK: startPos={startPos}")
            startQpos, jStart, startError = robustDLSik(model, data, obstacleIds, startPos)
            self.logger.debug(f"Post Start IK: startQpos={startQpos}, jStart={jStart}")

            goalQpos, jGoal, goalError = robustDLSik(model, data, obstacleIds, goalPos, initialQpos=startQpos)
            self.logger.debug(f"Post Goal IK: goalQpos={goalQpos}, jGoal={jGoal}")
            # startQpos = np.array([0.2, -0.8, -0.3, 0.9])
            # goalQpos = np.array([-0.4, 0.7, 0.5, -1.0])

            # if startQpos is None or goalQpos is None or startError > 0.1 or goalError > 0.1:
            if startQpos is None or goalQpos is None:
                reward = -100.0
                break

            self.logger.debug(f"Goal Error: {goalError}")
            self.logger.debug(f"Start Error: {startError}")

            if jStart.shape[1] == numLinks:
                muStart = manipulabilityIndex(jStart)
            else:
                muStart = 0.0
            self.logger.debug(f"Mu Start: {muStart}")
            if jGoal.shape[1] == numLinks:
                muGoal = manipulabilityIndex(jGoal)
            else:
                muGoal = 0.0
            self.logger.debug(f"Mu Goal: {muGoal}")
           
            foundSolution, path = rrtConnect(
                model,
                data,
                startQpos,
                goalQpos,
                obstacleIds,
                totalTime=1.0,
                stepSize=0.1,
                numIsteps=5,
                tol=0.01,
            )

            if foundSolution:
                numStates = len(path)
                qPoses = [np.array(q) for q in path]
                energyCost = 0.0
                eePathLength = 0.0

                totalTime = 0.0
                data.qpos[:] = qPoses[0]
                mujoco.mj_forward(model, data)
                prevEE = data.site("endEffector").xpos.copy()
                minDT = 0.001
                maxDT = 10.0
                for s in range(1, numStates):
                    data.qpos[:] = qPoses[s]
                    mujoco.mj_forward(model, data)
                    currEE = data.site("endEffector").xpos.copy()
                    eePathLength += np.linalg.norm(currEE - prevEE)
                    prevEE = currEE

                    q1 = qPoses[s - 1]
                    q2 = qPoses[s]
                    deltaQ = q2 - q1

                    lowDT = minDT
                    highDT = maxDT
                    feasibleDT = highDT

                    for _ in range(20):
                        midDT = (lowDT + highDT) / 2.0
                        v = deltaQ / midDT
                        qMid = (q1 + q2) / 2
                        data.qpos[:] = qMid
                        data.qvel[:] = v
                        data.qacc[:] = 0  # Assume constant vel for tau estimate
                        mujoco.mj_inverse(model, data)
                        tau = data.qfrc_inverse[:numLinks].copy()
                        if np.all(np.abs(tau) <= np.abs(model.actuator_ctrlrange[:, 1])):
                            feasibleDT = midDT  # Feasible, try smaller dt (faster)
                            highDT = midDT
                        else:
                            lowDT = midDT  # Too fast, increase dt

                    dt = feasibleDT
                    totalTime += dt
                    v = deltaQ / dt  # Updated realistic velocity

                    data.qpos[:] = qMid 
                    data.qvel[:] = v
                    data.qacc[:] = 0
                    mujoco.mj_inverse(model, data)
                    tau = data.qfrc_inverse[:numLinks].copy()

                    # Now compute power with this v
                    power = np.sum(np.abs(tau * v))  # Reuse tau from last inverse
                    energyCost += power * dt
                    # print(f"Step {s}: avg |tau| = {np.mean(np.abs(tau))}, avg |dq| = {np.mean(np.abs(v))}, dt = {dt}, power = {power}")
                self.logger.debug(f"Energy Cost: {energyCost}")
                self.logger.debug(f"End Effector Path Length: {eePathLength}")
            
                # print(f"Path Length Penalty: {-0.1 * eePathLength}")
                # print(f"Accuracy Penalty: {-100 * (startError + goalError)}")
                # print(f"Manipulability Bonus: {1 * (muStart + muGoal)}")
                # print(f"Link Number Penalty: {-0.1 * (numLinks - self.minNumLinks)}")
                # print(f"Energy Cost Penalty: {-0.001 * energyCost}")
                reward += 100 - 0.1 * eePathLength - 100 * (startError + goalError) + 1 * (muStart + muGoal) - 0.1 * (numLinks - self.minNumLinks) - 0.001 * energyCost
            else:
                # print(f"Accuracy Penalty: {-100 * (startError + goalError)}")
                # print(f"Manipulability Bonus: {1 * (muStart + muGoal)}")
                # print(f"Link Number Penalty: {-0.1 * (numLinks - self.minNumLinks)}")
                reward += 30 - 100 * (startError + goalError) + 1 * (muStart + muGoal) - 0.1 * (numLinks - self.minNumLinks)
                # reward += (
                #     30
                #     - 200 * (startError + goalError)
                #     - 1 * (numLinks - self.minNumLinks)
                # )

        avgReward = reward / len(self.startPos)
        self.logger.debug(f"Average reward: {avgReward}")

        return avgReward

    def step(self, action):
        # PPO GENERATED
        numLinks = int(np.round(action[0] * (self.maxNumLinks - self.minNumLinks) + self.minNumLinks))
        lengths = (action[1:(self.maxNumLinks + 1)] * (self.maxLength - self.minLength) + self.minLength)[:numLinks]
        jointTypes = np.round(action[(1 + self.maxNumLinks):] * 3)[:numLinks].astype(int)
        # # TEST
        # numLinks = 2
        # lengths = np.array([0.77285725, 1.1999999])
        # jointTypes = np.array([1, 0])
        # # PANDA
        # numLinks = 7
        # sizeMultiplier = 1
        # lengths = sizeMultiplier * np.array([0.333, 0.316, 0.0825, 0.0825, 0.384, 0.088, 0.01])
        # jointTypes = np.array([2, 1, 2, 0, 2, 0, 2])

        # print("Num Links: ", numLinks)
        # print("Lengths: ", lengths)
        # print("Joint Types: ", jointTypes)

        reward = self._evaluate(numLinks, lengths, jointTypes)
        done = True

        return np.array([0.0], dtype=np.float32), reward, done, done, {}

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

def generateXML(numJoints, lengths, jointTypes, taskConfig):
    try:
        # Barebones XML setup optimized for fast physics/kinematics
        xml = f"""
<mujoco>
    <compiler angle="radian" />
    <option gravity="0 0 -9.81" />
    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" />
        """

        # Inject dynamic obstacles (Physics only, no colors)
        for obs in taskConfig["obstacles"]:
            xml += f'<geom name="{obs["name"]}" type="box" pos="{obs["pos"]}" size="{obs["size"]}" />\n'

        # Inject Robot Base
        xml += f"""
        <body name="base" pos="{taskConfig['basePos']}" euler="{taskConfig['baseEuler']}">
            <geom name="baseBox" type="box" size="0.12 0.12 0.06" />
        """

        currentPos = "0 0 0.06"
        hingeLimit = 7 * math.pi / 8 
        
        for i in range(numJoints):
            li = lengths[i]
            jCode = jointTypes[i]
            
            if jCode == 0:
                axis, jtype, limitStr = "1 0 0", "hinge", f'limited="true" range="{-hingeLimit:.4f} {hingeLimit:.4f}"'
            elif jCode == 1:
                axis, jtype, limitStr = "0 1 0", "hinge", f'limited="true" range="{-hingeLimit:.4f} {hingeLimit:.4f}"'
            elif jCode == 2:
                axis, jtype, limitStr = "0 0 1", "hinge", 'limited="false"'
            elif jCode == 3:
                axis, jtype, limitStr = "0 0 1", "slide", f'limited="true" range="0 {li:.4f}"'
            
            xml += f"""
            <body name="link{i}" pos="{currentPos}">
                <joint name="joint{i}" type="{jtype}" axis="{axis}" damping="1.0" {limitStr} />
                <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {li:.4f}" />
            """
            currentPos = f"0 0 {li:.4f}"
            
        # End Effector Site
        xml += f'<site name="endEffector" pos="{currentPos}" size="0.015" />'
            
        xml += "</body>\n" * numJoints
        xml += "</body>\n" # Close the base body

        xml += "</worldbody>\n<actuator>\n"
        for i in range(numJoints):
            xml += f'<motor name="motor{i}" joint="joint{i}" ctrlrange="-10 10"/>\n'
        xml += "</actuator>\n</mujoco>"
        
        return xml
    except Exception as e:
        print(f"Mujoco XML Generation Error: {e}")
        raise

def manipulabilityIndex(J):
    if J is None or J.shape[0] != 6 or not np.all(np.isfinite(J)):
        return 0.0

    Sigma = np.linalg.svd(J, compute_uv=False)
    return np.min(Sigma)

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

# ────────────
# IK function 
# ────────────
def dlsIK(
    model,
    data,
    obstacleIds,
    targetPose: np.ndarray,
    initialQpos: np.ndarray | None = None,
    maxIter: int = 200,
    tol: float = 0.005,
    lambda_: float = 0.05,
    alpha: float = 0.75,
    rotWeight: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
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
    deltaX = np.concatenate([posError, rotError * rotWeight])

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
        deltaX = np.concatenate([posError, rotError * rotWeight])
        i += 1

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
    lambda_: float = 0.05,
    alpha: float = 1.0,
    numTries: int = 50,
    rotWeight: float = 0.2,
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
            data,
            obstacleIds,
            targetPose,
            initialQpos=initQ,
            maxIter=maxIter,
            tol=tol,
            lambda_=lambda_,
            alpha=alpha,
            rotWeight=rotWeight,
        )

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

def setupLogging():
    pid = os.getpid()
    logger = logging.getLogger(f"process{pid}")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"logs/logProcess{pid}.txt")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger