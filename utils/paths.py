"""Resolve artifact directories and discover files from explicit input patterns."""
from __future__ import annotations

import glob
from pathlib import Path
import re

from .config import AnalysisConfig, Dataset, DatasetConfig


def discover_files(dataset: Dataset) -> tuple[str, ...]:
    """List inputs in configured run order and numeric Nano-file order."""
    files = []
    for source in dataset.sources:
        matches = [p for p in glob.glob(source.input_pattern) if Path(p).is_file()]
        if not matches:
            raise FileNotFoundError(f"No Nano files for Run{source.run}: {source.input_pattern}")
        if len({Path(p).parent for p in matches}) != 1:
            raise ValueError(f"Run{source.run} matches multiple production directories; pin nano_production or narrow input_pattern")

        def file_order(path: str) -> tuple:
            name = Path(path).name
            suffix = re.fullmatch(rf"NANO_{source.run}_(\d+)\.root", name)
            return (0, int(suffix.group(1))) if suffix else (1, name)

        files.extend(sorted(matches, key=file_order))
    if len(files) != len(set(files)):
        raise ValueError("The input patterns select the same Nano file more than once")
    return tuple(files)


def resolve_paths(config: AnalysisConfig) -> dict:
    """Resolve shared artifacts and the distinct correction/target datasets."""
    from .identity import input_id, linear_id, population, variant
    from .datasets import calibration_label
    correction = config.correction
    target = config.target
    evaluation = (config.output_dir / target.campaign / "evaluation" / str(target.dataset.run)
                  / calibration_label(target) / target.selection / config.model_dir.name / variant(population(config, target=True)))

    def dataset_paths(recipe: DatasetConfig, diagnostics: Path) -> dict:
        campaign_root = config.output_dir / recipe.campaign
        return {
            "analysis_inputs": str(campaign_root / "prepared"),
            "prepared_runs": [str(campaign_root / "prepared" / f"Run{source.run}")
                              for source in recipe.dataset.sources],
            "calibration_file": str(recipe.calibration_file),
            "pedestal_constants": str(recipe.calibration_file.parent),
            "covariances": str(diagnostics / "covariances"),
            "noise_model_fits": str(diagnostics / "noise_model_fits"),
            "plots": str(Path(__file__).resolve().parents[1] / "plots"
                         / recipe.campaign / f"Run{recipe.dataset.run}"),
        }

    training = (config.output_dir / correction.campaign / "training"
                / str(correction.dataset.run) / calibration_label(correction)
                / correction.selection)
    training /= input_id(config)
    if config.derivations:
        training=config.output_dir/'joint'/'training'/input_id(config)
    # Both estimators share membership/count files; feature views live in metadata.
    return {
        "output_root": str(config.output_dir),
        "correction": dataset_paths(correction, config.model_dir / "diagnostics"),
        "target": {
            **dataset_paths(target, evaluation),
            "predictions": str(evaluation / "predictions"),
            "nano_exports": str(evaluation / "nano_exports"),
        },
        "training_inputs": str(training),
        "dnn_training_inputs": str(training),
        "analytic_predictors": str(config.model_dir / 'analytic' / linear_id(config)),
        "dnn_models": str(config.model_dir),
    }


def application_plot_directory(config: AnalysisConfig, directory: Path) -> Path:
    """Separate figure sets by exact target and selected model identities."""
    from .identity import population, variant
    from .inference_config import method_directory
    return directory/'variants'/variant({'target':population(config,target=True),
        'methods':{method: str(method_directory(config, method)) for method in config.methods if method != 'uncorrected'}})


def module_plot_directory(config: AnalysisConfig, module: str) -> Path:
    """Local plots follow the reference campaign/run/module/calibration layout."""
    from .datasets import calibration_label
    return application_plot_directory(config, Path(resolve_paths(config)["target"]["plots"]) / module
            / (calibration_label(config.target) if config.target.conditions else f"pedestals_from_Run{config.target.pedestal.run}")
            / config.model_dir.name / config.target.selection)
