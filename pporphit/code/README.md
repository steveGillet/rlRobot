<<<<<<< HEAD
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
=======
# PPOrphIt: PPO Robot Manipulator Morphology Design

This repository contains the source code for optimizing robot manipulator morphologies using PPO in a MuJoCo simulation environment, as described in the project paper.

## Project Overview
- **Goal**: Use reinforcement learning (PPO) to optimize robot arm designs (links, joints) for pick-and-place tasks.
- **Key Components**:
  - Dynamic MuJoCo XML generation for morphologies.
  - Damped least-squares IK and OMPL RRTConnect for planning.
  - Reward function based on success, path length, energy cost, etc.
- **Tasks**: Simple, Shelf, Container pick-and-place scenarios.

## Installation Instructions

### Prerequisites
- Python 3.12+
- OS: Linux/Mac (MuJoCo/OMPL may have Windows limitations)
- Install MuJoCo: Follow [official docs](https://mujoco.readthedocs.io/en/stable/python.html).
- Install OMPL: On Ubuntu, `sudo apt install libompl-dev`; for Python bindings, build from source or use `pip install pyompl` if available.

### Setup Environment
1. Clone the repo (or unzip this package).
2. Create a virtual environment: `python -m venv env` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
   - Note: If OMPL bindings fail, install manually (e.g., `pip install ompl` or from source).
4. (Optional) For Conda: `conda env create -f environment.yml` and activate.

## Usage

### Training the PPO Model
Run the training script:
```bash
python pporphiMain.py
>>>>>>> refs/remotes/origin/master
