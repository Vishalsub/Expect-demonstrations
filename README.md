# Pushing Environment and Expert Demonstration Export



## Description
This module sets up a custom MuJoCo-based robotic pushing environment for generating high-quality expert demonstrations using a pre-trained PPO agent. The environment simulates a ball-pushing task with configurable reward types and observation spaces.

### Features
- Custom MuJoCo XML environment for robotic ball pushing.
- PPO-based expert policy with deterministic action selection.
- High-quality demonstration filtering using reward percentile.
- Includes comprehensive unit tests.

## Directory Structure
```
.
├── push.py               # Environment setup (PushingBallEnv)
├── test.py               # Exports expert demonstrations to .npz
├── push_unittest.py      # Unit tests for the MuJoCo environment
├── assets/
│   └── pushxml/push.xml  # XML file defining the MuJoCo model
```

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```
Ensure `mujoco` is correctly installed and licensed.

### Run Expert Demonstration Export
```bash
python test.py
```
Output: `f_high_quality_demonstrations_1.npz`

### Run Unit Tests
```bash
python -m unittest push_unittest.py
```

## To-do
- [ ] Enable randomized ball/hole positions
- [ ] Add sparse reward mode
- [ ] Connect with LOKI learner module
- [ ] Visualize demonstration trajectories

> 💡 **NOTE:** Demonstration quality is filtered based on top 25% reward percentile.

## Contributing
Thank you for considering contributing to this project! Please check out the [Contributing Guidelines](/CONTRIBUTING.md).

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

## Issues
To report a bug or request a feature, please open an [Issue](https://github.com/your-username/pushing_env_demo/issues).

## Pull Requests
We welcome Pull Requests! Please follow our [Pull Request Template](/PULL_REQUEST_TEMPLATE.md) when submitting one.
