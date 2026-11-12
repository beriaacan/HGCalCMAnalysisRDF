"""Known step names and validation for an explicitly ordered execution plan."""
from __future__ import annotations

STEP_NAMES = (
    "pedestals", "prepare", "analytic", "training_inputs",
    "analytic_allinputs", "training_cache", "train", "submit_train", "apply", "plots", "export",
    "multimodule_plot",
)
# Updated as each processing step is implemented. Preflight checks this before
# any selected step runs, so an incomplete plan cannot start writing outputs.
IMPLEMENTED_STEPS: frozenset[str] = frozenset({
    "pedestals", "prepare", "training_inputs", "analytic", "analytic_allinputs", "training_cache", "train", "submit_train", "apply", "plots", "multimodule_plot", "export",
})


def validate_steps(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(step, str) for step in value):
        raise ValueError("steps must be a non-empty ordered list of step names")
    unknown = [step for step in value if step not in STEP_NAMES]
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}; choose from {', '.join(STEP_NAMES)}")
    if len(value) != len(set(value)):
        raise ValueError("steps must not contain duplicates")
    return tuple(value)
