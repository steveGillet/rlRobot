import mujoco
import mujoco.viewer
import time
import numpy as np
import ompl.base as ob
import ompl.geometric as og
from scipy.optimize import minimize
import gymnasium as gym
from ompl import util as ou
ou.setLogLevel(ou.LOG_NONE)

class robotArmEnv(gym.Env):
    # metadata = {"render_modes": []}

    def __init__(self, numLinks=4, fixedJointTypes=None):
        super().__init__()
        self.numLinks = numLinks
        self.action_space = gym.spaces.Box(low=0.05, high=1.2, shape=(numLinks,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

        self.fixedJointTypes = fixedJointTypes or [2,0,2,1]
        self.startPos = np.array([-.4, -0.4, 0.6], dtype=np.float32)
        self.goalPos = np.array([0.4, 0.4, 0.8], dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.array([0.0], dtype=np.float32), {}

    def _evaluate(self, lengths):
        try:
            xml = generateXML(self.numLinks, lengths.tolist(), self.fixedJointTypes)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
        except:
            return -50.0

        actuatorIds = [model.actuator(f"motor{i}").id for i in range(self.numLinks)]
        jointIds = [model.joint(f"joint{i}").id for i in range(self.numLinks)]

        space = ob.CompoundStateSpace()
        for link in range(self.numLinks):
            space.addSubspace(ob.SO2StateSpace(), 1.0)

        si = ob.SpaceInformation(space)
        simpleSetup = og.SimpleSetup(si)

        startQpos = ik_dls(model, self.startPos)
        goalQpos = ik_dls(model, self.goalPos, q_init=startQpos)
        # startQpos = np.array([0.2, -0.8, -0.3, 0.9])
        # goalQpos = np.array([-0.4, 0.7, 0.5, -1.0])
        
        if startQpos is None:
            return -100.0

        if goalQpos is None:
            return -100.0


        i = 0
        for id in jointIds:
            data.qpos[id] = goalQpos[i]
            i+=1

        mujoco.mj_forward(model, data)
        goalError = np.linalg.norm(data.site('endEffector').xpos - self.goalPos)

        i = 0
        for id in jointIds:
            data.qpos[id] = startQpos[i]
            i+=1

        mujoco.mj_forward(model, data)
        startError = np.linalg.norm(data.site('endEffector').xpos - self.startPos)

        start = ob.State(space)
        goal = ob.State(space)
        for i in range(len(startQpos)):
            start()[i].value = startQpos[i]
            goal()[i].value = goalQpos[i]
            
        simpleSetup.setStartAndGoalStates(start, goal)

        planner = og.RRTstar(si)
        simpleSetup.setPlanner(planner)
        simpleSetup.solve(0.8)
        planner.clear()

        foundSolution = simpleSetup.haveSolutionPath()

        if foundSolution:
            simpleSetup.simplifySolution()
            path = simpleSetup.getSolutionPath()
            length = path.length()
            path.clear()
            # path.interpolate(20)
            # pathStates = [path.getState(i) for i in range(path.getStateCount())]
            return 100 - 0.8 * length - 20 * (startError + goalError)

        else:
            # pathStates = []
            return 30 - 50 * (startError + goalError)
        
    def step(self, action):
        lengths = np.clip(action, 0.05, 1.2)
        reward = self._evaluate(lengths)
        done = True
        return np.array([0.0], dtype=np.float32), reward, done, done, {}

def ik_dls(
    model,
    target_pos: np.ndarray,
    q_init: np.ndarray | None = None,
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
    if q_init is None:
        q = np.zeros(nq, dtype=np.float64)
    else:
        q = np.array(q_init, dtype=np.float64).copy()
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
            return None

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
        return None

    return q_best


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
    
    res = minimize(objective, initialQpos, bounds=bounds, method='L-BFGS-B', options={'maxiter': maxIter, 'ftol': tol, 'disp': False})

    # print(res)

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