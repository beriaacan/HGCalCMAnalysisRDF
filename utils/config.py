"""Load human-editable YAML through documented defaults and strict validation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import yaml

from .calibrations import validate_calibration_run
from .steps import validate_steps
from .plot_config import KEYS as PLOT_KEYS, parse as parse_plots
from .train_config import KEYS as TRAIN_KEYS, parse as parse_training
from .inference_config import KEYS as APPLY_KEYS, parse as parse_application
from .multimodule_config import KEYS as MULTIMODULE_KEYS, parse as parse_multimodule

CAMPAIGNS = ("Jun2026TB", "Cassette")
METHODS = ("uncorrected", "analytic", "analytic_allinputs", "dnn")

from .config_defaults import SETTINGS, complete_settings


@dataclass(frozen=True)
class SourceRun:
    run: int
    input_pattern: str


@dataclass(frozen=True)
class Dataset:
    run: int | str
    sources: tuple[SourceRun, ...]
    adcmin: float | None
    adcmax: float | None


@dataclass(frozen=True)
class DatasetConfig:
    campaign: str
    dataset: Dataset
    pedestal: Dataset
    selection: str
    modules: str | tuple[str, ...]
    module_map: Path
    calibration_file: Path
    trigger_time_window: tuple[float, float]


@dataclass(frozen=True)
class AnalysisConfig:
    config_file: Path
    steps: tuple[str, ...]
    correction: DatasetConfig
    target: DatasetConfig
    output_dir: Path
    model_dir: Path
    methods: tuple[str, ...]
    dnn: dict[str, Any]
    resources: dict[str, int]
    diagnostics: dict[str, Any]
    pedestal_options: dict[str, bool]
    preparation: dict[str, Any]
    plotting: dict[str, Any]
    training: dict[str, Any]
    application: dict[str, Any]
    multimodule: dict[str, Any]


def local_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a path")
    expanded = Path(os.path.expandvars(value)).expanduser()
    return Path(os.path.abspath(expanded if expanded.is_absolute() else base / expanded))


def positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def module_values(value: Any, name: str) -> str | tuple[str, ...]:
    if isinstance(value, str) and value in {"all", "ld", "hd"}:
        return value
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be all, ld, hd, or an ordered module list")
    if any(not isinstance(m, str) or not m or not all(c.isalnum() or c == '_' for c in m) for m in value):
        raise ValueError("Module names must contain letters, numbers and underscores")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(value)


def dataset_value(runs: Any, pattern: Any, lower: Any, upper: Any, base: Path) -> Dataset:
    if type(runs) is int:
        runs = [runs]
    if not isinstance(runs, list) or not runs:
        raise ValueError("Run settings must contain a run number or non-empty run list")
    for run in runs:
        positive_int(run, "run")
    if len(runs) != len(set(runs)):
        raise ValueError("Run lists must not contain duplicates")
    if not isinstance(pattern, str) or "{run}" not in pattern:
        raise ValueError("input_pattern must contain {run}, including for pedestal discovery")
    sources = []
    for run in runs:
        try:
            expanded = pattern.format(run=run)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"Invalid input_pattern; only {{run}} is supported: {exc}") from None
        sources.append(SourceRun(run, str(local_path(expanded, base, "input_pattern"))))
    lower = number(lower, "ADC minimum") if lower is not None else None
    upper = number(upper, "ADC maximum") if upper is not None else None
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("ADC minimum must be smaller than ADC maximum")
    label = "_".join(str(run) for run in runs)
    if lower is not None:
        label += f"_adcmin{lower:g}"
    if upper is not None:
        label += f"_adcmax{upper:g}"
    key = runs[0] if len(runs) == 1 and lower is None and upper is None else label
    return Dataset(key, tuple(sources), lower, upper)


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject accidental duplicate settings instead of silently taking the last."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError('YAML setting names must be strings')
        if key in result:
            raise ValueError(f'Duplicate YAML setting {key!r} at line {key_node.start_mark.line + 1}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_settings(filename: str | Path) -> tuple[dict, dict]:
    """Return written and effective settings without opening detector inputs."""
    path = Path(filename).expanduser().resolve()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Use a .yaml or .yml setup file")
    try:
        with path.open(encoding="utf-8") as handle:
            settings = yaml.load(handle, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from None
    if not isinstance(settings, dict) or any(not isinstance(k, str) for k in settings):
        raise ValueError("The YAML setup must contain derivation/application mappings and workflow settings")
    written = settings
    settings = complete_settings(written, path.parent, local_path)
    if set(settings) != SETTINGS:
        raise RuntimeError('Configuration defaults/schema are inconsistent')
    return written, settings


def load_config(filename: str | Path, steps: list[str] | None = None) -> AnalysisConfig:
    path = Path(filename).expanduser().resolve()
    _, settings = read_settings(path)
    # Keep the established numerical validators/resources behind the public blocks.
    effective=settings
    source,application=settings['derivation'],settings['application']
    settings=dict(settings)
    settings.update({key:source[key] for key in ('split_seed','validation_split_fraction',
                                                'train_event_fraction','validation_event_fraction')})
    settings.update({f'apply_{key}':application[key] for key in
                     ('device','batch_events','events_per_chunk','prefetch_batches','pin_memory')})
    settings['infer_batch']=application['infer_batch']
    settings['allow_unknown_modules']=application['allow_unknown_modules']
    nulls=[key for key in SETTINGS if key!='condor_proxy' and effective[key] is None]
    if nulls:
        raise ValueError('These settings cannot be null: '+', '.join(sorted(nulls)))
    configured_steps=validate_steps(settings['steps'])
    selected_steps=configured_steps if steps is None else validate_steps(steps)

    def recipe(role):
        row=effective[role]
        campaign=row['campaign']
        dataset=dataset_value(row['runs'],row['input_pattern'],row['adc_min'],row['adc_max'],path.parent)
        pedestal_run=positive_int(row['pedestal_run'],f'{role}.pedestal_run')
        pedestal=dataset_value(pedestal_run,row['pedestal_input_pattern'],None,None,path.parent)
        selection=row['selection']
        if selection not in {'full','trigtime','test','test_trigtime'}:
            raise ValueError(f'{role}.selection must be full or trigtime; held-out selectors are not implemented')
        modules=row['modules']
        if role=='application' and modules=='all' and isinstance(source['modules'],list):
            modules=source['modules']
        modules=module_values(modules,f'{role}.modules')
        module_map=local_path(row['module_map'],path.parent,f'{role}.module_map')
        calibration_file=local_path(row['calibration_file'],path.parent,f'{role}.calibration_file')
        try:
            validate_calibration_run(calibration_file,pedestal_run)
        except ValueError as exc:
            raise ValueError(f'{role}: {exc}') from None
        window=tuple(number(row[name],f'{role}.{name}') for name in ('trigger_time_min','trigger_time_max'))
        if window[0]>=window[1]:
            raise ValueError(f'{role}: trigger_time_min must be smaller than trigger_time_max')
        return DatasetConfig(campaign,dataset,pedestal,selection,modules,module_map,calibration_file,window)

    correction=recipe('derivation')
    target=recipe('application')
    output = local_path(settings["output_dir"], path.parent, "output_dir")
    models = local_path(settings["model_dir"], path.parent, "model_dir")
    methods = settings["methods"]
    if not isinstance(methods, list) or not methods or any(m not in METHODS for m in methods):
        raise ValueError(f"methods must be a non-empty list drawn from {METHODS}")
    if len(methods) != len(set(methods)):
        raise ValueError("methods must not contain duplicates")
    loss = settings["loss"]
    if loss != "mse":
        raise ValueError("Only loss: mse is enabled for now; censored training is deferred")
    nodes = settings["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("nodes must be a non-empty list of positive integers")
    for width in nodes:
        positive_int(width, "node count")
    dropout = number(settings["dropout"], "dropout")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    unknown = settings['allow_unknown_modules']
    if type(unknown) is not bool:
        raise ValueError('allow_unknown_modules must be true or false')
    # Internal markers distinguish DNN physical features from linear statistics;
    # these are fixed contracts, never user preprocessing switches.
    dnn = {'preprocessing_mode': 'calibration_only', 'observation_model': loss,
           'nodes': nodes, 'dropout': dropout, 'preprocess_inputs': False,
           'preprocess_targets': False, 'allow_unknown_modules': unknown}
    for name in ("epochs", "batch_events", "infer_batch", "shuffle_buffer_chunks"):
        dnn[name] = positive_int(settings[name], name)
    for name in ("train_event_fraction", "validation_event_fraction"):
        fraction = number(settings[name], name)
        if not 0 < fraction <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
        dnn[name] = fraction
    dnn["learning_rate"] = number(settings["learning_rate"], "learning_rate")
    dnn["weight_decay"] = number(settings["weight_decay"], "weight_decay")
    if dnn["learning_rate"] <= 0 or dnn["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    dnn["sample_weighting"] = settings["sample_weighting"]
    if dnn["sample_weighting"] not in {"none", "source_run_channel"}:
        raise ValueError("sample_weighting must be none or source_run_channel")
    dnn['split_seed'] = settings['split_seed']
    if type(dnn['split_seed']) is not int or not 0 <= dnn['split_seed'] < 2**64:
        raise ValueError('split_seed must be an unsigned 64-bit integer')
    dnn['validation_split_fraction'] = number(settings['validation_split_fraction'], 'validation_split_fraction')
    if not 0 < dnn['validation_split_fraction'] < 1:
        raise ValueError('validation_split_fraction must be in (0, 1)')
    groups = settings['input_features']
    global_groups = {'cm_erx', 'activity_fractions', 'module_id', 'module_fraction', 'adc_allchannels'}
    local_groups = {'channel_in_erx', 'erx_index', 'cell_area_fraction'}
    if (not isinstance(groups, list) or not groups or any(not isinstance(g, str) for g in groups)
            or len(set(groups)) != len(groups) or set(groups) - global_groups - local_groups):
        raise ValueError('input_features must be a nonempty list of distinct supported feature groups: '
                         + ', '.join(sorted(global_groups | local_groups)))
    seen_local = False
    for group in groups:
        seen_local |= group in local_groups
        if seen_local and group in global_groups:
            raise ValueError('Place per-channel input_features after the event-level groups')
    dnn['input_features'] = tuple(groups)
    dnn['max_channels'] = positive_int(settings['max_channels'], 'max_channels')
    if dnn['max_channels'] != 444 or 'module_fraction' not in groups:
        raise ValueError('DNN inputs require max_channels: 444 and the module_fraction feature (Nchannels / 444)')
    dnn['missing_input_value'] = number(settings['missing_input_value'], 'missing_input_value')
    if abs(dnn['missing_input_value']) > 3.4028234663852886e38:
        raise ValueError('missing_input_value must be representable as float32')
    dnn['cell_area_file'] = local_path(settings['cell_area_file'], path.parent, 'cell_area_file')
    dnn['module_registry_file'] = local_path(settings['module_registry_file'], path.parent, 'module_registry_file')
    resources = {name: positive_int(settings[name], name)
                 for name in ("cpus", "events_per_chunk")}
    plot_inputs = settings["plot_dnninputs"]
    if type(plot_inputs) is not bool:
        raise ValueError("plot_dnninputs must be true or false")
    diagnostics = {"plot_dnninputs": plot_inputs,
                   "n_coherent_noise_model": positive_int(settings["n_coherent_noise_model"], "n_coherent_noise_model"),
                   "analytic_rcond": number(settings["analytic_rcond"], "analytic_rcond")}
    if diagnostics["analytic_rcond"] <= 0:
        raise ValueError("analytic_rcond must be positive")
    optimize_binning = settings["pedestal_optimize_binning"]
    if type(optimize_binning) is not bool:
        raise ValueError("pedestal_optimize_binning must be true or false")
    thresholds = settings["activity_thresholds"]
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("activity_thresholds must be a non-empty positive-integer list")
    for value in thresholds:
        positive_int(value, "activity threshold")
    if thresholds != sorted(set(thresholds)):
        raise ValueError("activity_thresholds must be increasing without duplicates")
    return AnalysisConfig(path, selected_steps, correction, target, output, models,
                          tuple(methods), dnn, resources, diagnostics,
                          {"optimize_binning": optimize_binning},
                          {"activity_thresholds": tuple(thresholds)}, parse_plots(settings),
                          parse_training(settings,path.parent,local_path), parse_application(settings,path.parent,local_path),
                          parse_multimodule(settings,path.parent,local_path))
