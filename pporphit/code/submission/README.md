# Robot Morphology Co-Design Ablation Study

This repository contains the code and trained models for the Robot Morphology Co-Design paper. It includes a custom MuJoCo-based environment (`robotArmEnv`), trained PPO agents, and a script to reproduce the ablation study and analytical plots presented in the paper.

## 1. Installation

### Dependencies
This codebase requires Python 3.8+ and the following libraries:
- `mujoco`
- `gymnasium`
- `numpy`
- `matplotlib`
- `torch`
- `stable_baselines3`
- `scipy`

- **Conda (Recommended for Mac/Linux):**
  ```
  conda install -c conda-forge ompl
  ```

### Python Package Installation
You can install the Python dependencies using pip:
```
pip install -r requirements.txt
```

## 2. File Structure
- `evaluate.py`: **Main Entry Point.** Runs the evaluation loops, calculates metrics, and generates the plots.
- `robotArmEnv.py`: Defines the Gymnasium environment, handles XML generation for various robot morphologies, and performs OMPL path planning.
- `simple_rl.py`: Contains the `REINFORCE` baseline training code. .
- `*.zip`: Pre-trained PPO model checkpoints (e.g., `shelfPreManArm.zip`, `twoShelfArmDoubleRew.zip`).
- `reinforce_policy.pth`: Pre-trained REINFORCE baseline weights.

## 3. Reproducing Results

To generate the ablation study plots and print the quantitative results:

1. Ensure you are in the directory containing the scripts and `.zip` models.
2. Run the ablation study script:

```
python evaluate.py
```

### Outputs
The script will evaluate the models on three tasks (Container, Shelf, TwoShelf) and output the following in the current directory:

1.  **Quantitative Metrics:** Printed to the console (Rewards, Success Rates, etc.).
2.  **Summary Plots:**
    - `ablation_Container_Task.png`
    - `ablation_Shelf_Task.png`
    - `ablation_TwoShelf_Task.png`

### Visualization
To watch the robot solve the tasks in MuJoCo:

```
python visualize_robot.py shelfPreManArm.zip
```

or on mac

```
mjpython visualize_robot.py shelfPreManArm.zip
```

This will load the model, generate the robot XML, plan paths for the tasks, and open a MuJoCo viewer to display the simulation.


## 4. Debugging

```
(env) jayvakil@Jays-Mac-mini ~/code/rlRobot/pporphit/code/submission $ python evaluate.py
Traceback (most recent call last):
  File "/Users/jayvakil/code/rlRobot/pporphit/code/submission/evaluate.py", line 208, in <module>
    main()
  File "/Users/jayvakil/code/rlRobot/pporphit/code/submission/evaluate.py", line 155, in main
    env = robotArmEnv()
  File "/Users/jayvakil/code/rlRobot/pporphit/code/submission/robotArmEnv.py", line 46, in __init__
    self.logger = setupLogging()
  File "/Users/jayvakil/code/rlRobot/pporphit/code/submission/robotArmEnv.py", line 591, in setupLogging
    handler = logging.FileHandler(f"logs/logProcess{pid}.txt")
  File "/Volumes/Crucial/miniconda/envs/env/lib/python3.10/logging/__init__.py", line 1169, in __init__
    StreamHandler.__init__(self, self._open())
  File "/Volumes/Crucial/miniconda/envs/env/lib/python3.10/logging/__init__.py", line 1201, in _open
    return open_func(self.baseFilename, self.mode,
FileNotFoundError: [Errno 2] No such file or directory: '/Users/jayvakil/code/rlRobot/pporphit/code/submission/logs/logProcess86610.txt'
```

To mitigate this error run this before running the `evaluate.py` script:

```
mkdir logs
```