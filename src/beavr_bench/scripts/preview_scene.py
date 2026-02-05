"""CLI script for previewing beavr-bench scenes.

Example:
    beavr-preview --scene scene_pickplace
"""

import argparse
import logging

from beavr_bench.engine.physics_loop import PhysicsLoop
from beavr_bench.envs.tasks import make_teleop_env

logger = logging.getLogger(__name__)


def main():
    """Preview a pre-baked scene."""
    parser = argparse.ArgumentParser(description="Preview a beavr-bench scene.")
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_pickplace",
        help="Name of the scene (e.g., scene_pickplace)",
    )
    parser.add_argument(
        "--rate-hz",
        type=int,
        default=30,
        help="Target physics rate in Hz",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading scene: {args.scene}")
    env = make_teleop_env(scene_name=args.scene)
    env.reset()

    logger.info("Starting PhysicsLoop...")
    loop = PhysicsLoop(task=env, rate_hz=args.rate_hz)

    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.stop()
        env.close()


if __name__ == "__main__":
    main()
