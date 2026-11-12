#!/usr/bin/env python3
"""Run the HGCal common-mode analysis steps selected in a flat YAML setup."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path

from utils.calibrations import validate_calibrations
from utils.config import load_config, read_settings
from utils.preprocessing import dnn_definition
from utils.paths import resolve_paths
from utils.steps import IMPLEMENTED_STEPS, STEP_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configured HGCal CM analysis steps.")
    parser.add_argument("--config", required=True, type=Path, help="YAML derivation/application recipe.")
    parser.add_argument("--steps", nargs="+", choices=STEP_NAMES,
                        help="Replace the YAML steps list; execute in the supplied order.")
    parser.add_argument("--show", action="store_true",
                        help="Show settings, paths and selected steps without processing.")
    parser.add_argument("--list-models", action="store_true",
                        help="List saved model references and training modules without running steps.")
    # Internal worker routing for submit_train's one-job-per-model submission.
    # Keep it out of the public interface; ordinary runs use the YAML grouping.
    parser.add_argument('--training-group', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = load_config(args.config, steps=args.steps)
        if args.training_group is not None:
            if config.steps != ('train',): parser.error('--training-group requires --steps train')
            from dataclasses import replace
            config=replace(config,training=dict(config.training,_selected_group=args.training_group))
        if args.list_models:
            if args.steps is not None or args.show:
                parser.error('--list-models cannot be combined with --steps or --show')
            from utils.model_catalogue import print_catalogue
            print_catalogue(config)
            return 0
        if args.show:
            written, effective = read_settings(args.config)
            print(json.dumps({
                "settings_supplied": sorted(written),
                "settings_resolved_or_defaulted": sorted(set(effective) - set(written)),
                "effective_settings": effective,
                "dnn_preprocessing": dnn_definition(),
                "configuration": asdict(config),
                "resolved_paths": resolve_paths(config),
                "calibrations": validate_calibrations(config, require_files=False),
                "pedestal_production": (importlib.import_module("pedestals").describe(config)
                                        if "pedestals" in config.steps else []),
                "preparation": (importlib.import_module("prepare").describe(config)
                                if "prepare" in config.steps else []),
                "training_input_preparation": (
                    importlib.import_module("training_inputs").describe(config)
                    if "training_inputs" in config.steps else None),
                "numeric_training_cache": (
                    importlib.import_module("training_cache").describe(config)
                    if "training_cache" in config.steps else None),
                "analytic": (importlib.import_module("analytic").describe(config)
                             if "analytic" in config.steps else None),
                "analytic_allinputs": (
                    importlib.import_module("analytic_allinputs").describe(config)
                    if "analytic_allinputs" in config.steps else None),
                "application": (importlib.import_module("application_step").describe(config)
                                if "apply" in config.steps else None),
                "dnn_training": (importlib.import_module("train").describe(config)
                                 if "train" in config.steps else None),
                "training_submission": (importlib.import_module("submit_train").describe(config)
                                        if "submit_train" in config.steps else None),
                "plotting": (importlib.import_module("plots").describe(config)
                             if "plots" in config.steps else None),
                "multimodule_plotting": (importlib.import_module("multimodule_plot").describe(config)
                                        if "multimodule_plot" in config.steps else None),
                "nano_export": (importlib.import_module("export").describe(config)
                                if "export" in config.steps else None),
                "requested_steps": config.steps,
                "available_steps": STEP_NAMES,
                "implemented_steps": [name for name in STEP_NAMES if name in IMPLEMENTED_STEPS],
            }, indent=2, default=str))
            return 0
        missing = [step for step in config.steps if step not in IMPLEMENTED_STEPS]
        if missing:
            parser.error("Processing steps are not implemented yet: " + ", ".join(missing)
                         + ". Add --show to inspect the selected plan.")
        if 'submit_train' in config.steps and config.steps[-1]!='submit_train':
            parser.error('submit_train must be the final step: submission does not wait for trained models')
        if 'submit_train' in config.steps and 'train' in config.steps:
            parser.error('Choose train or submit_train for one invocation')
        if config.steps[0] != "pedestals":
            validate_calibrations(config,require_files=True)
        for step in config.steps:
            from utils.workflow_registry import stage_guard
            with stage_guard(config, step):
                importlib.import_module("application_step" if step == "apply" else step).run(config)
            if step == "pedestals":
                validate_calibrations(config, require_files=True)
    except (OSError, ValueError, TypeError, NotImplementedError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
