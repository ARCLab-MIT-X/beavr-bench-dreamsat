"""Training entry point for mjlab-based RL training.

This module provides the CLI interface for launching RL training
using mjlab's ManagerBasedRlEnv with the teleop IL configuration.

Usage:
    # Train with default settings
    uv run python -m beavr_sim.train

    # Train with custom settings
    uv run python -m beavr_sim.train --num-envs 2048 --max-iterations 1000

    # Use with mjlab's train CLI (if registered)
    uv run train Beavr-Teleop-IL --env.scene.num-envs 1024
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import draccus

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for RL training."""

    # Environment settings
    num_envs: int = 1024
    """Number of parallel environments."""

    episode_length_s: float = 10.0
    """Episode length in seconds."""

    decimation: int = 4
    """Simulation steps per environment step."""

    # Training settings
    max_iterations: int = 10000
    """Maximum training iterations."""

    checkpoint_interval: int = 500
    """Save checkpoint every N iterations."""

    log_interval: int = 10
    """Log metrics every N iterations."""

    # Paths
    log_dir: str = "logs"
    """Directory for logs and checkpoints."""

    demo_path: str | None = None
    """Path to demonstration data for IL (.npz file)."""

    # Hardware
    device: str = "cuda:0"
    """Device for training."""

    seed: int = 42
    """Random seed."""


def setup_logging(level: int = logging.INFO):
    """Configure logging for training."""
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_training(config: TrainConfig):
    """Run RL training with the given configuration."""
    setup_logging()

    logger.info("🚀 Starting Beavr-Sim RL Training")
    logger.info(f"📊 Num envs: {config.num_envs}")
    logger.info(f"💾 Log dir: {config.log_dir}")
    logger.info(f"🎲 Seed: {config.seed}")

    # Import mjlab components
    try:
        from beavr_sim.envs.rl_env_cfg import TeleopILEnvCfg
        from beavr_sim.envs.scene_cfg import get_default_scene_cfg
        from mjlab.envs import ManagerBasedRlEnv
    except ImportError as e:
        logger.error(f"❌ mjlab not installed: {e}")
        logger.error("Install with: uv add mjlab")
        return

    # Create environment configuration
    logger.info("📦 Creating environment configuration...")

    scene_cfg = get_default_scene_cfg(num_envs=config.num_envs)

    env_cfg = TeleopILEnvCfg(
        decimation=config.decimation,
        episode_length_s=config.episode_length_s,
        seed=config.seed,
    )
    env_cfg.scene = scene_cfg

    # Create environment
    logger.info("🌍 Initializing environment...")
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=config.device,
        render_mode=None,
    )

    logger.info("✅ Environment created:")
    logger.info(f"   Observation space: {env.observation_space}")
    logger.info(f"   Action space: {env.action_space}")
    logger.info(f"   Num envs: {env.num_envs}")

    # TODO: Add actual training loop with algorithm
    # For now, just run a few steps to verify the environment works
    logger.info("🔄 Running environment sanity check...")

    obs, info = env.reset()
    logger.info(f"   Initial obs shape: {obs['policy'].shape}")

    import torch

    for i in range(10):
        # Random actions
        action = env.action_space.sample()
        # Convert to torch tensor
        action_torch = torch.from_numpy(action).to(env.device)
        obs, reward, terminated, truncated, info = env.step(action_torch)

        if i == 0:
            logger.info(f"   Step obs shape: {obs['policy'].shape}")
            logger.info(f"   Reward shape: {reward.shape}")

    logger.info("✅ Environment sanity check passed!")
    logger.info("")
    logger.info("🎯 Next steps:")
    logger.info("   1. Load demonstration data from --demo-path")
    logger.info("   2. Implement BC/GAIL/etc algorithm")
    logger.info("   3. Add training loop")

    env.close()
    logger.info("🏁 Training script complete")


def main():
    """Main entry point for RL training.

    Examples:
        # Basic training
        python -m beavr_sim.train

        # Custom settings
        python -m beavr_sim.train --num-envs 2048 --max-iterations 5000
    """
    cfg = draccus.parse(config_class=TrainConfig)
    run_training(cfg)


if __name__ == "__main__":
    main()
