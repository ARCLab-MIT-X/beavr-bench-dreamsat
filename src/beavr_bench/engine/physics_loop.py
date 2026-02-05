from __future__ import annotations

import logging
import time

import mujoco

from beavr_bench.envs.tasks import BaseTask

logger = logging.getLogger(__name__)


class PhysicsLoop:
    """Distilled physics loop for scene preview and basic interaction.

    This class provides a minimal version of beavr-sim's PhysicsLoop,
    adapted for beavr-bench scenarios without networking.
    """

    def __init__(
        self,
        task: BaseTask,
        rate_hz: int = 30,
        use_viewer: bool = True,
    ):
        """Initialize physics loop.

        Args:
            task: The simulation task/environment
            rate_hz: Target physics rate in Hz
            use_viewer: Whether to sync with native viewer
        """
        self.task = task
        self.dt = 1.0 / rate_hz
        self.use_viewer = use_viewer
        self._shutdown = False
        self._rate_hz = rate_hz

        logger.info(f"PhysicsLoop initialized at {rate_hz} Hz (dt={self.dt:.4f}s)")
        if use_viewer:
            logger.info("Native viewer enabled")

    def run(self):
        """Run the main physics loop until shutdown."""
        logger.info("Physics loop starting...")

        # Initialize environment
        self.task.reset()

        if self.use_viewer and self.task.viewer is None:
            self.task.init_viewer(key_callback=self._key_callback)

        try:
            while not self._shutdown:
                loop_start = time.time()

                # Step simulation
                # In this minimal version, we don't have external actions,
                # so we just step physics.
                with self.task.lock:
                    mujoco.mj_step(self.task.model, self.task.data)

                # Render viewer
                if self.use_viewer:
                    self.task.render()

                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = self.dt - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self._shutdown = True
            logger.info("Physics loop stopped")

    def _key_callback(self, keycode: int):
        """Handle keyboard events from the viewer."""
        import glfw

        if keycode == glfw.KEY_BACKSPACE:
            logger.info("Resetting simulation...")
            self.task.reset()

        elif keycode == glfw.KEY_ESCAPE:
            self._shutdown = True

    def stop(self):
        """Request graceful shutdown."""
        self._shutdown = True
