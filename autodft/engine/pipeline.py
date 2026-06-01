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

    def tick(self) -> None:
        """Execute one complete pipeline cycle (8 steps).

        Each cycle runs inside a single database session/transaction.
        The session is committed at the end if all steps succeed.
        """
        logger.debug("--- Pipeline tick start ---")

        with get_session() as session:
            # Step 1: Update SLURM job statuses
            logger.debug("Step 1: Updating status of running jobs")
            update_running_jobs(session, self.scheduler)

            # Step 2: Process finished jobs (check QM output)
            logger.debug("Step 2: Processing finished jobs")
            process_finished_jobs(session, self.qm_engine)

            # Step 3: Update task statuses based on job results
            logger.debug("Step 3: Updating task statuses")
            update_task_statuses(
                session,
                max_attempts=self.settings.pipeline.max_attempts,
            )

            # Step 4: Start follow-up tasks for successful tasks
            logger.debug("Step 4: Starting follow-up tasks")
            start_followup_tasks(session, self.settings)

            # Step 5: Process new entrypoints from the queue
            logger.debug("Step 5: Processing new entrypoints")
            for _ in range(self.settings.pipeline.max_simultaneous_entrypoints):
                queue_len = self.scheduler.get_queue_length()
                if queue_len > self.settings.pipeline.max_queue_length:
                    logger.debug(
                        "Queue length %d exceeds max %d; stopping entrypoint processing",
                        queue_len,
                        self.settings.pipeline.max_queue_length,
                    )
                    break
                if not process_next_entrypoint(session, self.settings):
                    logger.debug("No more entrypoints to process")
                    break

            # Step 6: Create retry jobs for failed attempts + start new tasks
            logger.debug("Step 6: Creating retry jobs and starting new tasks")
            create_retry_jobs(session, self.settings, self.qm_engine)
            start_new_tasks(session, self.settings, self.qm_engine)

            # Step 7: Submit pending jobs to scheduler
            logger.debug("Step 7: Submitting pending jobs to scheduler")
            submit_pending_jobs(session, self.scheduler, self.settings)

            # Commit the entire cycle
            session.commit()

        logger.debug("--- Pipeline tick complete ---")
