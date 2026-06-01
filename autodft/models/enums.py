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
