"""Enumerations shared across models."""

from enum import Enum


class TaskType(str, Enum):
    confsearch = "confsearch"
    optimization = "optimization"
    singlepoint = "singlepoint"
    singlepoint_vert_ox = "singlepoint_vert_ox"
    singlepoint_vert_red = "singlepoint_vert_red"
    singlepoint_vert_spin_change = "singlepoint_vert_spin_change"
    singlepoint_nbo = "singlepoint_nbo"


class TaskStatus(str, Enum):
    created = "created"
    pending = "pending"
    successful = "successful"
    failed = "failed"


class SlurmStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


# `slurm_status` holds the raw sacct string, so these sets -- not the enum --
# decide what the pipeline does with a job.
#
# Transient: the job is not finished. Keep polling; do NOT parse its output.
# COMPLETING in particular is common (epilog, node drain) and can last
# minutes on a network filesystem.
TRANSIENT_SLURM_STATES = frozenset({
    "PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED",
    "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING",
    "STAGE_OUT", "STOPPED", "UNKNOWN",
})

# Terminal: the job is over, one way or another, and its output can be read.
TERMINAL_SLURM_STATES = frozenset({
    "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
    "SPECIAL_EXIT",
})
