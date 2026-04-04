import numpy as np
import pandas as pd
import time
import mujoco
from robotArmEnv import (robotArmEnv, TASK_REGISTRY, robustDLSik, 
                        omplRRTConnect, manipulabilityIndex, generateXML)


def evaluate_morphology(name, numLinks, lengths, jointTypes, 
                       task_name="wallMount", num_runs=20, noise=0.01):
    print(f"\n{'='*80}")
    print(f"Evaluating: {name} on task '{task_name}' ({num_runs} runs)")
    print(f"{'='*80}")

    taskConfig = TASK_REGISTRY[task_name]
    obstacleNames = ["floor"] + [obs["name"] for obs in taskConfig["obstacles"]]

    xml = generateXML(numLinks, lengths.tolist(), jointTypes.tolist(), taskConfig)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    obstacleIds = set(model.geom(name).id for name in obstacleNames)

    total_trials = num_runs * len(taskConfig["starts"])
    successful_paths = 0
    posErrors, rotErrors = [], []
    path_lengths, energy_costs = [], []
    manipulabilities = []
    ik_fail_count = 0
    rrt_fail_count = 0

    for run in range(num_runs):
        if run % 5 == 0:
            print(f"  Run {run}/{num_runs}...")

        for startPos, goalPos in zip(taskConfig["starts"], taskConfig["goals"]):
            # Noisy poses
            noisyStartPos = startPos[:3] + np.random.normal(0, noise, 3)
            noisyStartQuat = startPos[3:] + np.random.normal(0, noise, 4)
            noisyStartQuat /= np.linalg.norm(noisyStartQuat)
            startPose = np.concatenate([noisyStartPos, noisyStartQuat])

            noisyGoalPos = goalPos[:3] + np.random.normal(0, noise, 3)
            noisyGoalQuat = goalPos[3:] + np.random.normal(0, noise, 4)
            noisyGoalQuat /= np.linalg.norm(noisyGoalQuat)
            goalPose = np.concatenate([noisyGoalPos, noisyGoalQuat])

            # IK — increased robustness for debugging
            startQpos, jStart, startError, startPosErr, startRotErr = robustDLSik(
                model, data, obstacleIds, startPose, numTries=25, maxIter=50)
            goalQpos, jGoal, goalError, goalPosErr, goalRotErr = robustDLSik(
                model, data, obstacleIds, goalPose, initialQpos=startQpos, numTries=25, maxIter=50)

            posErrors.extend([startPosErr, goalPosErr])
            rotErrors.extend([startRotErr, goalRotErr])

            muStart = manipulabilityIndex(jStart) if jStart is not None else 0.0
            muGoal = manipulabilityIndex(jGoal) if jGoal is not None else 0.0
            manipulabilities.extend([muStart, muGoal])

            if startPosErr > 0.1 or goalPosErr > 0.1:
                ik_fail_count += 1
                continue

            # RRT — give it more time
            foundSolution, path = omplRRTConnect(
                model, data, startQpos, goalQpos, obstacleIds, totalTime=2.0)

            if foundSolution:
                successful_paths += 1
                qPoses = [np.array(q) for q in path]

                # Path length
                eePathLength = 0.0
                data.qpos[:] = qPoses[0]
                mujoco.mj_forward(model, data)
                prevEE = data.site("endEffector").xpos.copy()
                for q in qPoses[1:]:
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    currEE = data.site("endEffector").xpos.copy()
                    eePathLength += np.linalg.norm(currEE - prevEE)
                    prevEE = currEE
                path_lengths.append(eePathLength)

                # Energy (same as before)
                energyPath = qPoses[::4]
                if energyPath[-1] is not qPoses[-1]:
                    energyPath.append(qPoses[-1])

                energyCost = 0.0
                tauLimits = np.abs(model.actuator_ctrlrange[:, 1])
                prevDT = 0.1
                for s in range(1, len(energyPath)):
                    q1 = energyPath[s-1]
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
            else:
                rrt_fail_count += 1

    # Final stats
    success_rate = (successful_paths / total_trials) * 100 if total_trials > 0 else 0.0
    avg_pos_error = np.mean(posErrors) if posErrors else float('inf')
    avg_rot_error = np.mean(rotErrors) if rotErrors else float('inf')
    avg_path_length = np.mean(path_lengths) if path_lengths else float('inf')
    avg_energy = np.mean(energy_costs) if energy_costs else float('inf')
    avg_manipulability = np.mean(manipulabilities) if manipulabilities else 0.0

    print(f"\nRESULTS FOR: {name}")
    print("-" * 70)
    print(f"Links:              {numLinks}")
    print(f"Success Rate:       {success_rate:.1f}% ({successful_paths}/{total_trials})")
    print(f"Avg Pos Error:      {avg_pos_error:.4f} m")
    print(f"Avg Rot Error:      {avg_rot_error:.4f} rad")
    print(f"Avg Path Length:    {avg_path_length:.4f} m")
    print(f"Avg Energy:         {avg_energy:.4f} J")
    print(f"Avg Manipulability: {avg_manipulability:.4f}")
    print(f"IK failures:        {ik_fail_count}/{total_trials}")
    print(f"RRT failures:       {rrt_fail_count}/{total_trials - ik_fail_count}")
    print("-" * 70)

    return {
        "model_name": name,
        "n_links": numLinks,
        "success_rate": success_rate,
        "avg_pos_error": avg_pos_error,
        "avg_rot_error": avg_rot_error,
        "avg_path_length": avg_path_length,
        "avg_energy": avg_energy,
        "avg_manipulability": avg_manipulability,
        "task": task_name,
        "num_runs": num_runs
    }


if __name__ == "__main__":
    # CHANGE THIS TO TEST DIFFERENT TASKS
    task_to_test = "outreach"   # try "container" to see if it works better
    num_runs = 100                # keep low while debugging

    all_results = []

    # Morphology
    custom_n = 2
    custom_lengths = np.array([0.05,0.27123675])
    custom_joints = np.array([1,3])

    stats = evaluate_morphology("PPO", 
                                custom_n, custom_lengths, custom_joints,
                                task_name=task_to_test, num_runs=num_runs)
    all_results.append(stats)

    # # Baselines
    # baselines = [
    #     ("Baseline_FANUC", 6, np.array([0.165, 0.330, 0.08, 0.285, 0.05, 0.05]), np.array([2, 0, 0, 2, 0, 2])),
    #     ("Baseline_PANDA", 7, np.array([0.333, 0.316, 0.0825, 0.0825, 0.384, 0.107, 0.05]), np.array([2, 0, 2, 0, 2, 0, 2])),
    # ]

    # for bname, n, lengths, joints in baselines:
    #     stats = evaluate_morphology(bname, n, lengths, joints,
    #                                 task_name=task_to_test, num_runs=num_runs)
    #     all_results.append(stats)

    df = pd.DataFrame(all_results)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_name = f"evaluation_{task_to_test}_{num_runs}runs_{timestamp}.csv"
    df.to_csv(csv_name, index=False)

    print(f"\n✅ Done! Results saved to → {csv_name}")