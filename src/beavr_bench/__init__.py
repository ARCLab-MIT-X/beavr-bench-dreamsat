"""BEAVR-Bench: Standalone evaluation benchmark for BEAVR."""

import os
import platform

# Set EGL as the default MuJoCo rendering backend on Linux if not already set.
# This ensures robust offscreen rendering in headless environments.
if platform.system() == "Linux" and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


__version__ = "1.0.0"

# Auto-register environments with Gymnasium
from beavr_bench.registry import register_beavr_envs

register_beavr_envs()

# Import BeavrEnv to trigger LeRobot's registration
try:
    from beavr_bench.registry import BeavrEnv  # noqa: F401
except ImportError:
    pass  # LeRobot not installed
