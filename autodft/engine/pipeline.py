"""Main pipeline worker -- the forever loop that drives the DFT pipeline.

Instantiate :class:`PipelineWorker` with a :class:`Settings` and a
:class:`Scheduler`, then call :meth:`run_forever` (or :meth:`tick` for
a single cycle, useful in tests).
"""

from __future__ import annotations

import logging
import time

from autodft.config import Settings
from autodft.db import get_session
from autodft.engine.entrypoint_processor import process_next_entrypoint
from autodft.engine.scheduler import Scheduler
from autodft.engine.state_machine import (
    create_retry_jobs,
    process_finished_jobs,
    start_followup_tasks,
    start_new_tasks,
    submit_pending_jobs,
    update_running_jobs,
    update_task_statuses,
)
from autodft.qm.base import QMEngine

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Orchestrates the eight-step pipeline cycle.

    Args:
        settings: Application configuration.
        scheduler: Job scheduler implementation (SLURM or Local).
        qm_engine: Quantum-mechanics engine for input/output handling.
    """

    def __init__(
        self,
        settings: Settings,
        scheduler: Scheduler,
        qm_engine: QMEngine,
    ) -> None:
        self.settings = settings
        self.scheduler = scheduler
        self.qm_engine = qm_engine
        # Set by _entrypoint_step(); False once the queue is drained.
        self._last_entrypoint_processed = False

    def run_forever(self) -> None:
        """Main loop -- runs until interrupted (e.g. ``KeyboardInterrupt``).

        Each iteration calls :meth:`tick` and then sleeps for the
        configured interval.
        """
        logger.info(
            "Pipeline worker starting (interval=%ds)",
            self.settings.pipeline.loop_interval_seconds,
        )
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                logger.info("Pipeline worker interrupted -- shutting down")
                break
            except Exception:
                logger.exception("Error in pipeline tick")

            time.sleep(self.settings.pipeline.loop_interval_seconds)

    def _run_step(self, session, label: str, fn) -> bool:
        """Run one pipeline step, commit it, and contain its failures.

        Each step commits on its own so that progress is durable the moment
        it is made. Previously the whole tick shared one transaction that was
        committed only at the very end, which meant a single raising job --
        a truncated ORCA output, a missing ensemble file -- discarded every
        other molecule's progress from that tick and then raised again on the
        next tick, forever. Returns True if the step completed.
        """
        logger.debug("Step: %s", label)
        try:
            fn()
            session.commit()
            return True
        except Exception:  # noqa: BLE001 - one bad step must not stop the rest
            logger.exception("Pipeline step %r failed; rolling it back and continuing", label)
            session.rollback()
            return False

    def tick(self) -> None:
        """Execute one complete pipeline cycle.

        Steps are committed individually and are isolated from each other:
        a failure in one step is logged and rolled back, and the remaining
        steps still run.
        """
        logger.debug("--- Pipeline tick start ---")

        with get_session() as session:
            self._run_step(session, "1: update running job statuses",
                           lambda: update_running_jobs(session, self.scheduler))

            self._run_step(session, "2: process finished jobs",
                           lambda: process_finished_jobs(session, self.qm_engine))

            self._run_step(session, "3: update task statuses",
                           lambda: update_task_statuses(
                               session, max_attempts=self.settings.pipeline.max_attempts))

            self._run_step(session, "4: start follow-up tasks",
                           lambda: start_followup_tasks(session, self.settings))

            # Step 5: expand queued entrypoints. Each entrypoint commits
            # separately -- process_next_entrypoint() rolls the session back
            # when a SMILES can't be expanded, and without a commit per item
            # that rollback also discarded every entrypoint expanded before it
            # in the same tick (including their time_started marks, so they
            # were re-processed indefinitely).
            for _ in range(self.settings.pipeline.max_simultaneous_entrypoints):
                queue_len = self.scheduler.get_queue_length()
                if queue_len > self.settings.pipeline.max_queue_length:
                    logger.debug(
                        "Queue length %d exceeds max %d; stopping entrypoint processing",
                        queue_len,
                        self.settings.pipeline.max_queue_length,
                    )
                    break
                processed = self._run_step(
                    session, "5: process next entrypoint",
                    lambda: self._entrypoint_step(session),
                )
                if not processed or not self._last_entrypoint_processed:
                    break

            self._run_step(session, "6a: create retry jobs",
                           lambda: create_retry_jobs(session, self.settings, self.qm_engine))
            self._run_step(session, "6b: start new tasks",
                           lambda: start_new_tasks(session, self.settings, self.qm_engine))

            self._run_step(session, "7: submit pending jobs",
                           lambda: submit_pending_jobs(session, self.scheduler, self.settings))

        logger.debug("--- Pipeline tick complete ---")

    def _entrypoint_step(self, session) -> None:
        """Process one entrypoint, recording whether there was one to process."""
        self._last_entrypoint_processed = process_next_entrypoint(session, self.settings)
