"""BEAVR-Bench: Standalone evaluation benchmark for BEAVR."""

__version__ = "1.0.0"

# Auto-register environments with Gymnasium
from beavr_bench.registry import register_beavr_envs

register_beavr_envs()

# Import BeavrEnv to trigger LeRobot's registration
try:
    from beavr_bench.registry import BeavrEnv  # noqa: F401
except ImportError:
    pass  # LeRobot not installed
