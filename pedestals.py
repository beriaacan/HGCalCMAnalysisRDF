"""Produce our own calibration JSON using the central pedestal tools."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import termios

from utils.calibrations import validate_calibration_run, central_calibration_files
from utils.config import AnalysisConfig, Dataset, DatasetConfig
from utils.paths import discover_files
from utils.nano import read_module_metadata as _metadata, root_path as _root_path

FRAMEWORK_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PedestalJob:
    target: Path
    campaign: str
    pedestal: Dataset
    selections: tuple[str | tuple[str, ...], ...]
    roles: tuple[str, ...]


def _refuse_central_calibration(recipe: DatasetConfig, role: str) -> None:
    """Only produce local constants when the central campaign has none for this run."""
    from utils.config_defaults import CAMPAIGN_PATHS
    directory = FRAMEWORK_ROOT.parent.parent / CAMPAIGN_PATHS[recipe.campaign][2]
    block = {'derive': 'derivation', 'apply': 'application'}.get(role, role)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{block}: cannot check central calibrations: {directory} is missing. "
            "Set up the central Calibrations checkout before running pedestals."
        )
    central = central_calibration_files(directory, recipe.pedestal.run)
    if central:
        locations = "\n".join(f"  {path}" for path in central)
        raise ValueError(
            f"{block}: refusing pedestal production: a matching central calibration already "
            f"exists for Run{recipe.pedestal.run}:\n{locations}\n"
            "Use the existing central calibration and omit pedestals from steps. "
            f"If needed, set {block}.calibration_file to the path above."
        )



def plan(config: AnalysisConfig, *, for_execution: bool = False) -> tuple[PedestalJob, ...]:
    """Group shared targets; central calibrations remain read-only inputs."""
    recipes = (("derive", config.correction), ("apply", config.target))
    # Check central availability before reporting local-target configuration or
    # overwrite errors. Existing central inputs remain usable in mixed recipes
    # where another block genuinely requests a missing local calibration.
    local_targets = tuple((role, recipe) for role, recipe in recipes
                          if recipe.calibration_file.resolve().is_relative_to(
                              (FRAMEWORK_ROOT / "calibrations" / recipe.campaign).resolve()))
    missing_local = tuple((role, recipe) for role, recipe in local_targets
                          if not os.path.lexists(recipe.calibration_file))
    for role, recipe in missing_local or local_targets or recipes:
        _refuse_central_calibration(recipe, role)
    grouped: dict[Path, list[tuple[str, DatasetConfig]]] = {}
    available_local = set()
    for role, recipe in recipes:
        target = recipe.calibration_file.resolve()
        validate_calibration_run(target, recipe.pedestal.run)
        allowed = (FRAMEWORK_ROOT / "calibrations" / recipe.campaign).resolve()
        if not target.is_relative_to(allowed):
            if recipe.calibration_file.is_file():
                continue
            raise ValueError(f"{role}: pedestal production can only write inside {allowed}: {target}")
        if os.path.lexists(recipe.calibration_file):
            if not recipe.calibration_file.is_file():
                raise ValueError(f'Calibration target is not a regular file: {target}')
            available_local.add(target)
            continue
        grouped.setdefault(target, []).append((role, recipe))
    jobs = []
    for target, entries in grouped.items():
        first = entries[0][1]
        if any(recipe.pedestal != first.pedestal or recipe.campaign != first.campaign
               for _, recipe in entries):
            raise ValueError(f"Conflicting pedestal inputs for shared target: {target}")
        jobs.append(PedestalJob(target, first.campaign, first.pedestal,
                                tuple(dict.fromkeys(recipe.modules for _, recipe in entries)),
                                tuple(role for role, _ in entries)))
    if not jobs:
        if available_local:
            raise FileExistsError('No pedestal production needed; requested local calibrations '
                                  'already exist and will not be overwritten:\n'
                                  + '\n'.join(f'  {path}' for path in sorted(available_local)))
        raise ValueError('No missing local pedestal target. Omit calibration_file for automatic '
                         'central-or-local resolution, or choose a target under this framework\'s '
                         'calibrations/<campaign>/ directory. Existing constants are never overwritten.')
    return tuple(jobs)


def describe(config: AnalysisConfig) -> list[dict]:
    """Describe production without discovering or opening Nano files."""
    return [{"target": str(job.target), "pedestal_run": job.pedestal.run,
             "input_patterns": [source.input_pattern for source in job.pedestal.sources],
             "module_selections": job.selections, "roles": job.roles,
             "cpus": config.resources["cpus"],
             "optimize_binning": config.pedestal_options["optimize_binning"]}
            for job in plan(config)]


def _select_modules(job: PedestalJob, files: tuple[str, ...]) -> dict[str, tuple[int, int, int]]:
    available = _metadata(files[0])
    selected: dict[str, tuple[int, int, int]] = {}
    for selection in job.selections:
        if isinstance(selection, str):
            names = [name for name in available if selection == "all"
                     or (selection == "ld" and name.startswith("ML_"))
                     or (selection == "hd" and name.startswith("MH_"))]
        else:
            names = list(selection)
        if not names:
            raise ValueError(f"Module selection {selection} is empty for pedestal run {job.pedestal.run}")
        for name in names:
            if name not in available:
                raise ValueError(f"Module {name} absent from pedestal run {job.pedestal.run}")
            selected[name] = available[name]
    for filename in files[1:]:
        other = _metadata(filename)
        for name, identity in selected.items():
            if other.get(name) != identity:
                raise ValueError(f"Module metadata changes for {name}: {filename}")
    return selected


def _command(arguments: list[str], log_path: Path) -> None:
    """Tee terminal output without breaking ROOT's carriage-return progress bars."""
    print(f"Running {arguments[1]} (log: {log_path})", flush=True)
    master, slave = pty.openpty()
    try:
        try:
            columns, rows = os.get_terminal_size(sys.stdout.fileno())
        except (OSError, ValueError):
            columns, rows = 100, 24
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        with log_path.open("wb") as log:
            with subprocess.Popen(arguments, stdout=slave, stderr=slave) as process:
                os.close(slave)
                slave = None
                while True:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError as exc:
                        # Linux PTYs signal end-of-output with EIO once the slave closes.
                        if exc.errno == errno.EIO:
                            break
                        raise
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    if hasattr(sys.stdout, "buffer"):
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    else:
                        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
                returncode = process.wait()
    finally:
        os.close(master)
        if slave is not None:
            os.close(slave)
    if returncode:
        raise RuntimeError(f"Calibration command failed with exit status {returncode}; see {log_path}")


def _check_payload(path: Path, modules: dict[str, tuple[int, int, int]]) -> None:
    """Catch empty, incomplete or misassociated output before publishing it."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {name.replace("_", "-"): 37 * identity[2] for name, identity in modules.items()}
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError(f"Produced calibration does not cover exactly the requested modules: {path}")
    for name, nch in expected.items():
        row = payload[name]
        if not isinstance(row, dict):
            raise ValueError(f"Invalid calibration entry for {name} in {path}")
        for field in ("Channel", "Valid", "ADC_ped", "Noise", "CM_ped", "CM_slope"):
            if not isinstance(row.get(field), list) or len(row[field]) != nch:
                raise ValueError(f"Wrong {field} length for {name} in {path}")
        if row["Channel"] != list(range(nch)):
            raise ValueError(f"Unexpected channel order for {name} in {path}")


def _publish(source: Path, target: Path) -> None:
    """Publish a complete file without overwriting, including on AFS."""
    if os.path.lexists(target):
        raise FileExistsError(f"Refusing to overwrite calibration: {target}")
    # AFS disallows hard links between directories, even within one volume.
    # Stage in the target directory, then link there atomically without clobbering.
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.",
                                                 suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), source.stat().st_mode & 0o777)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def run(config: AnalysisConfig) -> None:
    """Run central derivation and conversion, then publish without overwriting."""
    jobs = plan(config, for_execution=True)
    scripts = FRAMEWORK_ROOT.parent.parent / "LocalCalibration" / "scripts"
    producer = scripts / "HGCalPedestals.py"
    converter = scripts / "PrepareLevel0CalibParams.py"
    for script in (producer, converter):
        if not script.is_file():
            raise FileNotFoundError(f"Central calibration tool is missing: {script}")
    # Resolve every job before creating output directories or starting a producer.
    prepared = []
    for job in jobs:
        files = discover_files(job.pedestal)
        for filename in files:
            if re.fullmatch(rf"NANO_{job.pedestal.run}_[0-9]+\.root", Path(filename).name) is None:
                raise ValueError(f"Nano filename does not declare pedestal run {job.pedestal.run}: {filename}")
        modules = _select_modules(job, files)
        prepared.append((job, files, modules))
    for job, files, modules in prepared:
        job.target.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f"{job.target.stem}.work-", dir=job.target.parent))
        print(f"Producing {job.target} from {len(files)} Nano file(s), {len(modules)} module(s)", flush=True)
        print(f"Temporary results and logs (removed after success): {work}", flush=True)
        scanmap = work / "scan_map.json"
        scanmap.write_text(json.dumps({"inc": {"idx": 0, "input": [_root_path(f) for f in files],
                                               "params": {}}}, indent=2) + "\n")
        manifest = {"campaign": job.campaign, "pedestal_run": job.pedestal.run,
                    "target": str(job.target), "files": files, "modules": modules,
                    "cpus": config.resources["cpus"], "pedestal_options": config.pedestal_options,
                    "central_tools": {str(script): hashlib.sha256(script.read_bytes()).hexdigest()
                                      for script in (producer, converter)}}
        (work / "production.json").write_text(json.dumps(manifest, indent=2) + "\n")
        central_output = work / "central"
        command = [sys.executable, str(producer), "--scanmap", str(scanmap),
                   "--output", str(central_output), "--moduleList", ",".join(modules),
                   "--maxThreads", str(config.resources["cpus"])]
        if config.pedestal_options["optimize_binning"]:
            command.append("--optimizeBinning")
        _command(command, work / "pedestals.log")
        pedestal_json = central_output / "pedestals.json"
        if not pedestal_json.is_file():
            raise FileNotFoundError(f"Central producer did not create {pedestal_json}")
        result = work / job.target.name
        _command([sys.executable, str(FRAMEWORK_ROOT / "utils" / "export_level0.py"),
                  "--central-scripts", str(scripts), "--pedestals", str(pedestal_json),
                  "--output", str(result), "--cpus", str(config.resources["cpus"])],
                 work / "export.log")
        _check_payload(result, modules)
        validate_calibration_run(job.target, job.pedestal.run)
        _publish(result, job.target)
        print(f"Calibration ready: {job.target}", flush=True)
        shutil.rmtree(work)
