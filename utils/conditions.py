"""Central Eras resolution with frozen run applicability and file identities.

The central helper owns database/IOV selection. This adapter locates equivalent
read-only files and pins them; it never creates calibration constants.
"""
from functools import lru_cache
import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if type(a) is float or type(b) is float:
        return (type(a) in (float, int) and type(b) in (float, int)
                and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-12))
    return a == b


@lru_cache(maxsize=16)
def central(era, run):
    from HGCalCommissioning.LocalCalibration.HGCalErasConfigDBHelper import HGCalErasConfigDBHelper
    # Central read-only helper performs era and latest-valid-run lookup.
    with contextlib.redirect_stdout(sys.stderr):
        return HGCalErasConfigDBHelper(database='Eras').getEraConfig(era, run)


def validate(record, calibration, module_map):
    if not isinstance(record, dict) or set(record) != {'era', 'run', 'resolved'}:
        raise ValueError('Frozen conditions require era, run and resolved provenance')
    frozen = record['resolved']
    if (not isinstance(frozen, dict) or frozen.get('schema') != 'central_era_files_v1'
            or frozen.get('era') != record['era'] or frozen.get('run') != record['run']
            or frozen.get('calibration_file') != str(calibration)
            or frozen.get('module_map') != str(module_map)
            or frozen.get('calibration_sha256') != file_hash(calibration)
            or frozen.get('module_map_sha256') != file_hash(module_map)
            or not isinstance(frozen.get('central_payload_sha256'), str)):
        raise ValueError('Frozen central conditions differ from their era/run or selected files')


def resolve(written, calibration, module_map, package, campaign, runs):
    if (not isinstance(written, dict) or set(written) - {'era', 'run', 'resolved'}
            or not isinstance(written.get('era'), str)
            or len(written['era'].split('/')) != 2
            or type(written.get('run')) is not int or written['run'] <= 0):
        raise ValueError('conditions must specify an explicit era (e.g. P5/v27) and positive run')
    if list(runs) != [written['run']]:
        raise ValueError('Each conditions block must select exactly its validated run; use separate dataset blocks')
    record = dict(written)
    if 'resolved' in record:
        if calibration is None or module_map is None:
            raise ValueError('Frozen conditions must contain explicit calibration_file and module_map')
        validate(record, calibration, module_map)
        return record, str(calibration), str(module_map)
    answer = central(record['era'], record['run'])
    payload = answer.get('modcalib')
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or not payload:
        raise ValueError('Central era has no usable RecHitCalib payload')
    folder = Path(package)/'Calibrations'/('P5' if campaign == 'Cassette' else 'TB2026')
    candidates = [Path(calibration)] if calibration is not None else sorted((folder/'calib').rglob('*.json'))
    matches = []
    for path in candidates:
        try:
            if equivalent(json.loads(path.read_text()), payload):
                matches.append(path)
        except (OSError, ValueError):
            continue
    if len(matches) != 1:
        raise ValueError(f'Central {record["era"]}/Run{record["run"]} requires one equivalent calibration file; '
                         f'found {len(matches)}. Specify calibration_file explicitly; no file is copied or guessed.')
    text = answer.get('modules')
    if not isinstance(text, str):
        raise ValueError('Central era has no module-map contents')
    if module_map is None:
        era_name = record['era'].replace('/', '')
        preferred = folder/'maps'/f'modulelocator_{era_name}.txt'
        maps = [preferred] if preferred.is_file() else sorted((folder/'maps').glob('modulelocator*.txt'))
        maps = [p for p in maps if p.read_text().strip() == text.strip()]
        if len(maps) != 1:
            raise ValueError('Specify module_map matching the selected central era; ambiguous/missing geometry')
        module_map = maps[0]
    if Path(module_map).read_text().strip() != text.strip():
        raise ValueError('module_map differs from the selected central era')
    calibration = matches[0]
    record['resolved'] = {'schema': 'central_era_files_v1', 'era': record['era'], 'run': record['run'],
        'central_payload_sha256': digest(payload), 'calibration_file': str(calibration),
        'calibration_sha256': file_hash(calibration), 'module_map': str(module_map),
        'module_map_sha256': file_hash(module_map),
        'agreement': 'recursive equality; floating serialization roundoff within 1e-14 relative/1e-12 absolute'}
    validate(record, calibration, module_map)
    return record, str(calibration), str(module_map)
