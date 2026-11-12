"""Central wafer polygons and validated Nano/locator detector placement."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from .analytic_artifacts import sha
from .nano import root_path
from .prepared import _locator

WAFER_PITCH_CM = 16.74408
IROT_OFFSET_DEG = 270.


def rotate(polygon, irot):
    if type(irot) is not int or irot not in range(6):
        raise ValueError(f'Detector-map irot must be in [0,5], found {irot}')
    angle = np.deg2rad(IROT_OFFSET_DEG+60.*irot)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(polygon) @ rotation.T


def module_center(u, v):
    return WAFER_PITCH_CM*np.asarray([u-.5*v, np.sqrt(3.)*.5*v])


def reflect(polygon, layer):
    result = np.array(polygon, copy=True)
    if layer % 2 == 0:
        result[..., 0] *= -1.
    return result


def central_polygons(config, modules):
    """Reuse fill_wafer_hist, including its CM-bin removal and channel order."""
    from HGCalCommissioning.LocalCalibration.plot.wafer import fill_wafer_hist
    folder = config.multimodule['multimodule_geometry_dir']
    by_type, sources = {}, {'central_wafer_helper': {
        'path': inspect.getsourcefile(fill_wafer_hist),
        'sha256': sha(Path(inspect.getsourcefile(fill_wafer_hist)))}}
    for name, info in modules.items():
        kind = name[:4]
        if kind in by_type:
            if len(by_type[kind]) != info['nch']:
                raise ValueError(f'Module geometry channel count differs within {kind}')
            continue
        path = folder/f'geometry_{kind}_wafer.root'
        if not path.is_file():
            raise FileNotFoundError(f'Missing central wafer template: {path}')
        # Check availability before central helper dereferences the file handle.
        histogram = fill_wafer_hist([0.]*info['nch'], modtype=kind, tempdir=str(folder))
        polygons = []
        for cell in histogram.GetBins():
            graph = cell.GetPolygon()
            polygon = np.asarray([(graph.GetPointX(i), graph.GetPointY(i)) for i in range(graph.GetN())])
            if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
                polygon = polygon[:-1]
            if len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError(f'Invalid central channel polygon: {kind}/{len(polygons)}')
            polygons.append(polygon)
        if len(polygons) != info['nch']:
            raise ValueError(f'Central geometry has {len(polygons)} channels for {kind}, expected {info["nch"]}')
        by_type[kind] = polygons
        sources[kind] = {'path': str(path), 'sha256': sha(path)}
    return by_type, sources


def positions(config, modules):
    """Check positions in every used Nano Runs tree against the YAML locator."""
    import ROOT
    locator = _locator(config.target.module_map)
    result, files = {}, {}
    for name, info in modules.items():
        row = locator[name]
        missing = {'plane', 'u', 'v', 'irot'}-row.keys()
        if missing:
            raise ValueError(f'Module locator lacks physical placement fields {sorted(missing)} for {name}')
        place = {'layer': int(row['plane']), 'u': int(row['u']), 'v': int(row['v']), 'irot': int(row['irot'])}
        if place['irot'] not in range(6):
            raise ValueError(f'Invalid locator irot for {name}')
        result[name] = place
        for source in info['sources']:
            manifest_path = Path(source['input_file']).parents[1]/'manifest.json'
            if manifest_path not in files:
                files[manifest_path] = json.loads(manifest_path.read_text())['definition']['files']
    used = {}
    for name, info in modules.items():
        for source in info['sources']:
            manifest_path = Path(source['input_file']).parents[1]/'manifest.json'
            original = next(row for row in files[manifest_path] if row['index'] == source['source_file_index'])
            used.setdefault(original['path'], {'source': original, 'modules': set()})['modules'].add(name)
    inventory = []
    for filename, item in used.items():
        handle = ROOT.TFile.Open(root_path(filename), 'READ')
        if not handle or handle.IsZombie():
            raise OSError(f'Cannot open Nano Runs metadata: {filename}')
        try:
            if str(handle.GetUUID().AsString()) != item['source']['uuid']:
                raise ValueError(f'Nano UUID changed while resolving detector placement: {filename}')
            tree = handle.Get('Runs')
            if not tree or int(tree.GetEntries()) != 1:
                raise ValueError(f'Expected exactly one Nano Runs row: {filename}')
            branches = {str(b.GetName()) for b in tree.GetListOfBranches()}
            needed = {'HGCReadout_Layer', 'HGCReadout_i1', 'HGCReadout_i2'} | {
                f'HGCTypeCodes_{name}' for name in item['modules']}
            if needed-branches:
                raise ValueError(f'Nano geometry metadata missing {sorted(needed-branches)}: {filename}')
            tree.GetEntry(0)
            arrays = [list(getattr(tree, key)) for key in ('HGCReadout_Layer', 'HGCReadout_i1', 'HGCReadout_i2')]
            if len({len(a) for a in arrays}) != 1:
                raise ValueError('Nano readout geometry arrays have different lengths')
            for name in item['modules']:
                indices = list(getattr(tree, f'HGCTypeCodes_{name}'))
                if len(indices) != 1 or not 0 <= int(indices[0]) < len(arrays[0]):
                    raise ValueError(f'Invalid Nano module index for {name}')
                actual = {key: int(a[int(indices[0])]) for key, a in zip(('layer', 'u', 'v'), arrays)}
                if actual != {key: result[name][key] for key in actual}:
                    raise ValueError(f'Nano/locator position mismatch for {name}: Nano={actual}, locator={result[name]}')
            inventory.append({'path': filename, 'uuid': item['source']['uuid'], 'modules': sorted(item['modules'])})
        finally:
            handle.Close()
    if config.target.campaign == 'Jun2026TB':
        layers = [p['layer'] for p in result.values()]
        if len(layers) != len(set(layers)):
            raise ValueError('June layer-ordered maps require at most one selected module per layer')
    else:
        coordinates = [(p['layer'], p['u'], p['v']) for p in result.values()]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError('Two selected modules occupy the same Cassette position')
    return result, {'locator': str(config.target.module_map), 'locator_sha256': sha(config.target.module_map),
                    'nano_runs': inventory, 'positions': result}


def finish(polygons, refs, outlines, annotations, detail, coordinates):
    if not polygons:
        raise ValueError('No connected channels in detector geometry')
    vertices = np.concatenate(polygons)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    padding = max(high-low)*.025
    if annotations: high[1] += .065*max(high-low)
    return {'polygons': polygons, 'refs': refs, 'outlines': outlines, 'annotations': annotations,
            'detail': detail, 'coordinates': coordinates, 'xlim': (low[0]-padding, high[0]+padding),
            'ylim': (low[1]-padding, high[1]+padding)}


def detector_geometries(config, modules, polygons, placement):
    from scipy.spatial import ConvexHull
    cassette = config.target.campaign == 'Cassette'
    names = sorted(modules, key=lambda n: (placement[n]['layer'], n))
    groups = [[n for n in names if placement[n]['layer'] == layer]
              for layer in sorted({p['layer'] for p in placement.values()})] if cassette else [names]
    geometries = []
    ncols = min(4, max(1, int(np.ceil(np.sqrt(len(names)*4./3.)))))
    for selected in groups:
        cells, refs, outlines, annotations = [], [], [], []
        for i, name in enumerate(selected):
            place = placement[name]
            center = module_center(place['u'], place['v']) if cassette else np.asarray(
                [(i % ncols)*1.20*WAFER_PITCH_CM, -(i//ncols)*1.30*WAFER_PITCH_CM])
            connected = np.asarray(modules[name]['area_fraction']) > 0
            wafer = []
            for channel in np.flatnonzero(connected):
                cell = rotate(polygons[name[:4]][channel], place['irot'])+center
                if cassette: cell = reflect(cell, place['layer'])
                wafer.append(cell); cells.append(cell); refs.append((name, int(channel)))
            if wafer:
                vertices = np.concatenate(wafer)
                hull = vertices[ConvexHull(vertices).vertices]
                outlines.append(np.vstack([hull, hull[0]]))
            if not cassette:
                annotations.append((center[0], center[1]+.60*WAFER_PITCH_CM,
                                    f'Layer {place["layer"]}  {name.rsplit("_", 1)[-1]}'))
        detail = f'Layer {placement[selected[0]]["layer"]}' if cassette else 'Layer-ordered stack'
        geometry = finish(cells, refs, outlines, annotations, detail, cassette)
        if not cassette:
            # The reference's label slot is free for its 11-module 4x3 stack.
            # A fully occupied smaller grid has no such slot; put labels below
            # the axes rather than obscuring a real wafer.
            geometry['label_position'] = (.765, .315) if len(names) % ncols else (.02, -.04)
        geometries.append(geometry)
    if cassette and len(geometries) > 1:
        vertices = np.concatenate([np.concatenate(g['polygons']) for g in geometries])
        low, high = vertices.min(axis=0), vertices.max(axis=0); padding = max(high-low)*.025
        for g in geometries:
            g['xlim'] = (low[0]-padding, high[0]+padding); g['ylim'] = (low[1]-padding, high[1]+padding)
    return geometries
