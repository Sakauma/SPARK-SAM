"""Experiment protocol guards and artifact-lineage helpers."""

from .reproduction import (
    ProtocolViolation,
    audit_auto_prompt_training_config,
    audit_spark_training_config,
    reproduction_protocol_enabled,
    finalize_prompt_teacher_lineage,
    resolve_initialization_checkpoint,
    validate_selection_lock,
    write_checkpoint_lineage,
)

__all__ = [
    "ProtocolViolation",
    "audit_auto_prompt_training_config",
    "audit_spark_training_config",
    "reproduction_protocol_enabled",
    "finalize_prompt_teacher_lineage",
    "resolve_initialization_checkpoint",
    "validate_selection_lock",
    "write_checkpoint_lineage",
]
