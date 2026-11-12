"""Check the run declared by a standard calibration filename before use."""
from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AnalysisConfig


def central_calibration_files(directory: Path, pedestal_run: int) -> list[Path]:
    """Share the pedestal producer's exact central-file discovery convention."""
    if not directory.is_dir():
        raise FileNotFoundError(f'Cannot check central calibrations: {directory} is missing. '
                                'Set up the central Calibrations checkout first.')
    return sorted(path for path in directory.rglob(f'level0_calib_{pedestal_run}.json') if path.is_file())


def default_calibration_file(central: Path, local: Path, pedestal_run: int) -> Path:
    """Prefer one central calibration; otherwise select the deterministic local target."""
    matches = central_calibration_files(central, pedestal_run)
    if len(matches) > 1:
        locations = '\n'.join(f'  {path}' for path in matches)
        raise ValueError(f'Multiple central calibrations for Run{pedestal_run}:\n{locations}\n'
                         'Set calibration_file explicitly to select one; no file is guessed.')
    return matches[0] if matches else local/f'level0_calib_{pedestal_run}.json'


def validate_calibration_run(path: Path, pedestal_run: int) -> None:
    """Require an explicit run in the filename, matching the selected pedestal."""
    match = re.fullmatch(r"level0_calib_([1-9][0-9]*)\.json", path.name)
    if match is None:
        raise ValueError(
            f"Calibration filename must be level0_calib_<run>.json: {path}. "
            "Save generic central exports under their source run's name."
        )
    source_run = int(match.group(1))
    if source_run != pedestal_run:
        raise ValueError(
            f"Calibration run mismatch: {path} declares run {source_run}, "
            f"but pedestal_run is {pedestal_run}"
        )


def validate_calibrations(config: AnalysisConfig, *, require_files: bool, roles=('derive', 'apply')) -> dict:
    """Report missing files for inspection; require them before consuming steps."""
    results = {}
    from .datasets import recipes
    for role, recipe in [*(("derive",r) for r in recipes(config)), ("apply", config.target)]:
        if role not in roles:
            continue
        path = recipe.calibration_file
        if recipe.conditions:
            from .conditions import validate
            validate(recipe.conditions,path,recipe.module_map)
        else:
            validate_calibration_run(path, recipe.pedestal.run)
        if path.exists() and not path.is_file():
            raise ValueError(f"{role} calibration_file is not a regular file: {path}")
        available = path.is_file()
        if require_files and not available:
            raise FileNotFoundError(f"Missing {role} calibration file: {path}")
        label=role if role not in results else f'{role}_{len(results)}'
        results[label] = {
            "file": str(path),
            "pedestal_run": recipe.pedestal.run,
            "filename_run_matches": None if recipe.conditions else True,
            "era_run_validated": bool(recipe.conditions),
            "status": "available" if available else "missing",
            "conditions": recipe.conditions,
        }
    return results
