# 🦫 BEAVR Bench

## GPU-Accelerated RL & Imitation Learning for Robotic Manipulation

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![CI](https://github.com/ARCLab-MIT/beavr-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/ARCLab-MIT/beavr-bench/actions/workflows/ci.yml)
[![GitHub Stars](https://img.shields.io/github/stars/ARCLab-MIT/beavr-bench?style=social)](https://github.com/ARCLab-MIT/beavr-bench)
[![arXiv](https://img.shields.io/badge/arXiv-2508.09606-b31b1b.svg)](https://arxiv.org/abs/2508.09606)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/collections/arclabmit/beavr-sim)

**BEAVR Bench** is a simulation suite designed for DreamSat test and evaluate physical AI algorithms.

It unifies state-of-the-art tools like **[MuJoCo](https://github.com/google-deepmind/mujoco)**, **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)**, **[Isaac Lab](https://github.com/mujocolab/mjlab)**, and **[LeRobot](https://github.com/huggingface/lerobot)** into a single, cohesive benchmarking platform for robotic learning.

Whether you are researching imitation learning, reinforcement learning, BEAVR Bench provides the performance needed to iterate quickly.

---

## 📑 Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Docker](#docker)
- [Usage](#usage)
- [Releases](#releases)
- [Getting Help](#getting-help)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)

---

## Key Features

- **GPU Acceleration**: Run parallel environments on a single GPU for massive throughput.
- **Unified API**: Seamless integration with **LeRobot** for training and evaluating policies.
- **Robot & Scene Support**: Robots and scenes are pre-built using the **MuJoCo Menagerie** library in **MuJoCo**.

---

## DreamSat Simulation Preview

<div align="center">
<table>
  <tr>
    <td align="center">
      <b>DreamSat Preview</b><br>
      <img src="media/dreamsat.png" width="400px">
    </td>
  </tr>
</table>
</div>

---

## Installation

### Installation Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA-capable GPU (recommended)

### Quick Start (Linux / macOS)

```bash
# 1. Clone the repository
git clone https://github.com/ARCLab-MIT/beavr-bench-dreamsat.git
cd beavr-bench-dreamsat

# 2. Install dependencies with uv
uv sync

# 3. Verify installation
uv run beavr-eval --help
```

### Quick Start (Windows)

```powershell
# 1. Install uv (Python package manager)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Close and reopen PowerShell (or run this to refresh PATH in current session)
$env:Path += ";$env:USERPROFILE\.local\bin"

# 3. Clone the repository
git clone https://github.com/ARCLab-MIT/beavr-bench-dreamsat.git
cd beavr-bench-dreamsat

# 4. Install Python 3.12 (required for PyTorch compatibility)
uv python install 3.12

# 5. Install dependencies using Python 3.12
uv sync --python 3.12

# 6. Verify installation
uv run beavr-eval --help
```

> **Note for Windows users:** This project requires Python 3.12. If you have a newer Python version (3.13+) installed globally, `uv` will automatically use the correct version for this project without affecting your system Python.

---

## Docker

BEAVR Bench can be run with Docker.

### Docker Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)

### Building the Image

```bash
docker build -t beavr-bench-dreamsat .
```

### Running with GPU Support

To run tests inside the container:

```bash
docker run --rm --gpus all \
    -e MUJOCO_GL=egl \
    beavr-bench-dreamsat
```

To run a specific script:

```bash
docker run --rm --gpus all \
    -e MUJOCO_GL=egl \
    beavr-bench-dreamsat uv run beavr-eval --help
```

### Windows Setup (Recommended)

BEAVR Bench runs on Windows through Docker with WSL2. This provides a complete Linux environment with GPU support.

#### Prerequisites

1. **Windows 10/11** (version 2004 or higher)
2. **NVIDIA GPU** with updated drivers
3. **Docker Desktop for Windows** - [Download here](https://docs.docker.com/desktop/install/windows-install/)
4. **WSL2** - Usually enabled automatically by Docker Desktop
5. **NVIDIA Container Toolkit** - [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Setup Steps

1. **Install Docker Desktop**:
   - Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - During installation, ensure "Use WSL 2 instead of Hyper-V" is selected
   - Restart your computer when prompted

2. **Enable WSL2 Integration**:
   - Open Docker Desktop
   - Go to Settings → Resources → WSL Integration
   - Enable integration with your default WSL distribution

3. **Install NVIDIA Container Toolkit**:

   ```powershell
   # In PowerShell as Administrator
   wsl --install
   wsl --set-default-version 2
   ```

   Then inside WSL2 (Ubuntu):

   ```bash
   # 1. Configure the NVIDIA stable repository
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # 2. Install the toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

   > **Important:** If you are using Docker Desktop, you do NOT need to run `sudo systemctl restart docker`. Instead, go to Docker Desktop **Settings > Resources > WSL Integration** and ensure "Ubuntu" is toggled **ON**, then click **Apply & Restart**.

4. **Verify GPU Access**:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

   You should see your GPU listed.

5. **Build and Run BEAVR Bench**:

   ```bash
   # Clone the repository (if not already done)
   git clone https://github.com/ARCLab-MIT/beavr-bench-dreamsat.git
   cd beavr-bench-dreamsat

   # Build the Docker image
   docker build -t beavr-bench-dreamsat .

   # Run tests
   docker run --rm --gpus all -e MUJOCO_GL=egl beavr-bench-dreamsat
   ```

---

## Usage

BEAVR Bench is fully compatible with LeRobot scripts.

### Training a Policy

```bash
beavr-train \
    --policy.type=act \
    --dataset.repo_id=arclabmit/xarm7_beavrsim_shellgame_dataset \
    --policy.repo_id=arclabmit/xarm7_act_beavrsim_shellgame_model \
    --dataset.video_backend=pyav \
    --env.type=beavr \
    --env.scene=scene_shellgame \
    --eval.batch_size=25 \
    --wandb.enable=true \
    --job_name=xarm7_act_beavrsim_shellgame_model
```

### Evaluating a Policy

```bash
beavr-eval \
    --policy.path=arclabmit/xarm7_act_beavrsim_shellgame_model \
    --env.scene=scene_shellgame \
    --eval.n_episodes=50
```

---

## Releases

This project uses automated releases via GitHub Actions.

### Triggering a Release

To create a new release:

1. Update the version in `pyproject.toml`.
2. Commit and push the change.
3. Create and push a new tag:

```bash
git tag v1.0.1
git push origin v1.0.1
```

The GitHub Action will automatically build the package and create a new GitHub Release with the artifacts attached.

---

## Contributing

We welcome contributions! Whether it's adding a new robot from the Menagerie, designing a new task, or fixing a bug.

Please check out our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting Help

Need assistance? Here's how to get support:

- **📖 Documentation**: Check out [SCENES.md](docs/SCENES.md) for detailed task descriptions.
- **🐛 Bug Reports**: [Open an issue](https://github.com/ARCLab-MIT-X/beavr-bench/issues) on GitHub.
- **📧 Contact**: Reach out to the maintainers at [alexposadas24@gmail.com](mailto:alexposadas24@gmail.com).

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## Citations

If you use BEAVR Bench in your research, please cite the following works:

### BEAVR Teleop

```bibtex
@misc{posadasnava2025beavr,
  title         = {BEAVR: Bimanual, multi-Embodiment, Accessible, Virtual Reality Teleoperation System for Robots},
  author        = {Alejandro Posadas-Nava and Alejandro Carrasco and Richard Linares},
  year          = {2025},
  eprint        = {2508.09606},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {Accepted for presentation at ICCR 2025, Kyoto},
  url           = {https://arxiv.org/abs/2508.09606}
}
```

---

<p align="center">
  Built with ❤️ at <a href="https://github.com/ARCLab-MIT"><b>MIT's ARCLab</b></a> for Space Robotics Research
</p>
