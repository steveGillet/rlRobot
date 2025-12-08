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