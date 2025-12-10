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

**Important:** This project also relies on the **Open Motion Planning Library (OMPL)** python bindings.
- **Conda (Recommended for Mac/Linux):**
  ```bash
  conda install -c conda-forge ompl
  ```

### Python Package Installation
You can install the Python dependencies using pip:
```bash
pip install -r requirements.txt
```

## 2. File Structure
- `ablation_study.py`: **Main Entry Point.** Runs the evaluation loops, calculates metrics, and generates the plots.
- `robotArmEnv.py`: Defines the Gymnasium environment, handles XML generation for various robot morphologies, and performs OMPL path planning.
- `simple_rl.py`: Contains the `PolicyNetwork` class definition (required to load the REINFORCE baseline).
- `*.zip`: Pre-trained PPO model checkpoints (e.g., `shelfPreManArm.zip`, `twoShelfArmDoubleRew.zip`).
- `reinforce_policy.pth`: Pre-trained REINFORCE baseline weights.

## 3. Reproducing Results

To generate the ablation study plots and print the quantitative results:

1. Ensure you are in the directory containing the scripts and `.zip` models.
2. Run the ablation study script:

```bash
python ablation_study.py
```

### Outputs
The script will evaluate the models on three tasks (Container, Shelf, TwoShelf) and output the following in the current directory:

1.  **Quantitative Metrics:** Printed to the console (Rewards, Success Rates, etc.).
2.  **Summary Plots:**
    - `ablation_Container_Task.png`
    - `ablation_Shelf_Task.png`
    - `ablation_TwoShelf_Task.png`
3.  **Analytical Plots:**
    - `analysis_tradeoff_[TaskName].png`: Visualizes the Complexity vs. Capability trade-off.
    - `analysis_manipulability_[TaskName].png`: Compares the Manipulability Index of agents.

## 4. Troubleshooting
- **OMPL Import Error:** Ensure OMPL is correctly installed in your python environment. If using Conda, check `conda list ompl`.
- **MuJoCo Rendering:** The scripts use `mujoco.viewer` or headless physics stepping. No external MuJoCo key is required for MuJoCo 2.1+.
