import numpy as np
import mujoco
import time

# Import your simulation helper functions from your main environment file
# (Change 'robotArmEnv' to whatever your main file is named)
from robotArmEnv import TASK_REGISTRY, generateXML, robustDLSik, rrtConnect, manipulabilityIndex

def evaluate_morphology(name, numLinks, lengths, jointTypes, task_name="wallMount", num_runs=100, noise=0.01):
    print(f"\n{'='*50}")
    print(f"Evaluating: {name} on task '{task_name}'")
    print(f"{'='*50}")
    
    taskConfig = TASK_REGISTRY[task_name]
    obstacleNames = ["floor"] + [obs["name"] for obs in taskConfig["obstacles"]]
    
    # Generate the model once for this morphology
    xml = generateXML(numLinks, lengths.tolist(), jointTypes.tolist(), taskConfig)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    obstacleIds = set([model.geom(name).id for name in obstacleNames])
    
    # Metric trackers
    total_trials = num_runs * len(taskConfig["starts"])
    successful_paths = 0
    posErrors = []
    rotErrors = []
    path_lengths = []
    energy_costs = []
    manipulabilities = []

    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Processing run {run}/{num_runs}...")
            
        for start_idx, (base_start, base_goal) in enumerate(zip(taskConfig["starts"], taskConfig["goals"])):
            
            # 1. Apply Gaussian Noise (sigma = 0.05 as per your paper)
            noisyStartPos = base_start[:3] + np.random.normal(0, noise, size=3)
            noisyStartQuat = base_start[3:] + np.random.normal(0, noise, size=4)
            noisyStartQuat /= np.linalg.norm(noisyStartQuat)
            startPos = np.concatenate([noisyStartPos, noisyStartQuat])
            
            noisyGoalPos = base_goal[:3] + np.random.normal(0, noise, size=3)
            noisyGoalQuat = base_goal[3:] + np.random.normal(0, noise, size=4)
            noisyGoalQuat /= np.linalg.norm(noisyGoalQuat)
            goalPos = np.concatenate([noisyGoalPos, noisyGoalQuat])

            # 2. Solve IK
            startQpos, jStart, startWeightedError, startPosErr, startRotErr = robustDLSik(model, data, obstacleIds, startPos, numTries=50, maxIter=200)
            goalQpos, jGoal, goalWeightedError, goalPosErr, goalRotErr = robustDLSik(model, data, obstacleIds, goalPos, initialQpos=startQpos, numTries=50, maxIter=200)
            
            posErrors.append(startPosErr)
            posErrors.append(goalPosErr)
            rotErrors.append(startRotErr)
            rotErrors.append(goalRotErr)
            
            muStart = manipulabilityIndex(jStart) if jStart.shape[1] == numLinks else 0.0
            muGoal = manipulabilityIndex(jGoal) if jGoal.shape[1] == numLinks else 0.0
            manipulabilities.append(muStart)
            manipulabilities.append(muGoal)

            # 3. Path Planning
            if startWeightedError > 0.1 or goalWeightedError > 0.1:
                continue # IK failed, skip RRT, count as failure

            foundSolution, path = rrtConnect(
                model, data, startQpos, goalQpos, obstacleIds,
                totalTime=3.0, stepSize=0.1, numIsteps=5, tol=0.01
            )

            if foundSolution:
                successful_paths += 1
                qPoses = [np.array(q) for q in path]
                
                # Calculate Path Length
                eePathLength = 0.0
                data.qpos[:] = qPoses[0]
                mujoco.mj_forward(model, data)
                prevEE = data.site("endEffector").xpos.copy()
                
                for s in range(1, len(qPoses)):
                    data.qpos[:] = qPoses[s]
                    mujoco.mj_forward(model, data)
                    currEE = data.site("endEffector").xpos.copy()
                    eePathLength += np.linalg.norm(currEE - prevEE)
                    prevEE = currEE
                path_lengths.append(eePathLength)

                # Calculate Energy
                energyPath = qPoses[::4]
                if energyPath[-1] is not qPoses[-1]:
                    energyPath.append(qPoses[-1])
                
                tauLimits = np.abs(model.actuator_ctrlrange[:, 1])
                prevDT = 0.1 
                energyCost = 0.0

                for s in range(1, len(energyPath)):
                    q1 = energyPath[s - 1]
                    q2 = energyPath[s]
                    deltaQ = q2 - q1
                    qMid = (q1 + q2) / 2.0
                    
                    lowDT, highDT = 0.001, prevDT * 5.0 
                    feasibleDT = highDT

                    for _ in range(7):
                        midDT = (lowDT + highDT) / 2.0
                        v = deltaQ / midDT
                        data.qpos[:] = qMid
                        data.qvel[:] = v
                        data.qacc[:] = 0 
                        mujoco.mj_inverse(model, data)
                        tau = np.abs(data.qfrc_inverse[:numLinks])
                        
                        if np.all(tau <= tauLimits):
                            feasibleDT = midDT  
                            highDT = midDT
                        else:
                            lowDT = midDT  

                    prevDT = feasibleDT
                    v = deltaQ / feasibleDT
                    data.qpos[:] = qMid 
                    data.qvel[:] = v
                    data.qacc[:] = 0
                    mujoco.mj_inverse(model, data)
                    tau = np.abs(data.qfrc_inverse[:numLinks])
                    energyCost += np.sum(tau * np.abs(v)) * feasibleDT
                    
                energy_costs.append(energyCost)

    # --- Print Results ---
    success_rate = (successful_paths / total_trials) * 100
    avg_pos_error = np.mean(posErrors) if posErrors else float('inf')
    avg_rot_error = np.mean(rotErrors) if rotErrors else float('inf')
    avg_path_length = np.mean(path_lengths) if path_lengths else float('inf')
    avg_energy = np.mean(energy_costs) if energy_costs else float('inf')
    avg_manipulability = np.mean(manipulabilities) if manipulabilities else 0.0

    print(f"\nRESULTS FOR: {name}")
    print("-" * 30)
    print(f"Total Links:        {numLinks}")
    print(f"Success Rate:       {success_rate:.1f}% ({successful_paths}/{total_trials})")
    print(f"Avg Position Error: {avg_pos_error:.4f} m")
    print(f"Avg Rotation Error: {avg_rot_error:.4f} rad")
    print(f"Avg Path Length:    {avg_path_length:.4f} m")
    print(f"Avg Energy Cost:    {avg_energy:.4f} J")
    print(f"Avg Manipulability: {avg_manipulability:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    task_to_test = "container" # Change this to test other tasks
    
    # 1. PMorph (Replace with your actual decoded output from rollout.py)
    # Example:
    pmorph_links = 2
    pmorph_lengths = np.array([0.05, 0.4981559])
    pmorph_types = np.array([0, 1])
    
    evaluate_morphology("PMorph Optimized", pmorph_links, pmorph_lengths, pmorph_types, task_name=task_to_test)

    # 2. Baseline: FANUC LR Mate 200iD (Parameters from your LaTeX draft)
    fanuc_links = 6
    fanuc_lengths = np.array([0.165, 0.330, 0.08, 0.285, 0.05, 0.05])
    fanuc_types = np.array([2, 0, 0, 2, 0, 2])
    
    evaluate_morphology("Baseline: FANUC", fanuc_links, fanuc_lengths, fanuc_types, task_name=task_to_test)

    # 3. Baseline: Franka Emika Panda (Parameters from your LaTeX draft)
    panda_links = 7
    panda_lengths = np.array([0.333, 0.316, 0.0825, 0.0825, 0.384, 0.107, 0.05])
    panda_types = np.array([2, 0, 2, 0, 2, 0, 2])
    
    evaluate_morphology("Baseline: PANDA", panda_links, panda_lengths, panda_types, task_name=task_to_test)