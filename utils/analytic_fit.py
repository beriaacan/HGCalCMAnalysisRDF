"""Compute one module's regressions, without deciding how results are published."""
from __future__ import annotations

import json
from pathlib import Path

from .analytic import candidate_columns, initialize_analytic_root, solve
from .nano import root_path
from .training import training_view


def _sources(manifest, module):
    for prepared in manifest['definition']['prepared']:
        if module not in prepared['modules']:
            continue
        for source in prepared['files']:
            item = next(row for row in prepared['outputs'] if row['module'] == module
                        and row['source_file_index'] == source['index'])
            yield (Path(prepared['directory']) / item['file'],
                   dict(source, run=prepared['run']))


def _expected_counts(manifest, module, partition):
    nch = manifest['feature_spec']['modules'][module]['nch']
    return [sum(row[partition][c] for row in manifest['counts_by_module_run'][module].values())
            for c in range(nch)]


def _view(ROOT, path, source, config, spec, module, partition):
    rdf = ROOT.RDataFrame('Events', root_path(str(path)))
    return training_view(rdf, config, spec, module, source).Filter(
        f'training_retained && training_split=={partition}', 'retained regression partition')


def compute_module(config, manifest, module: str, target: Path) -> tuple[dict, dict]:
    """Write coefficients and fit diagnostics; do not evaluate validation events."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    ROOT = initialize_analytic_root()
    if ROOT.IsImplicitMTEnabled() and ROOT.GetThreadPoolSize() != config.resources['cpus']:
        ROOT.DisableImplicitMT()
    if not ROOT.IsImplicitMTEnabled():
        ROOT.EnableImplicitMT(config.resources['cpus'])
    spec = manifest['feature_spec']
    meta = spec['modules'][module]
    columns = candidate_columns(spec, module)
    storage_mb = meta['nch']*len(columns)**2*8/1024**2
    print(f'Fitting {module}: {meta["nch"]} targets, {len(columns)} candidate predictors '
          f'({storage_mb:.0f} MiB covariance storage)', flush=True)
    fitter = ROOT.hgcal_cm_analytic.Fitter(ROOT.VecOps.RVec['int'](columns),
        meta['nch'], config.resources['events_per_chunk'])
    sources = list(_sources(manifest, module))
    for path, source in sources:
        print(f'  Training Run{source["run"]}/file{source["index"]}', flush=True)
        fitter.consume(ROOT.RDF.AsRNode(_view(ROOT, path, source, config, spec, module, 0)))
    counts = [int(fitter.count(c)) for c in range(meta['nch'])]
    if counts != _expected_counts(manifest, module, 'train'):
        raise ValueError(f'Reconstructed training counts differ from training_inputs: {module}')
    with threadpool_limits(limits=config.resources['cpus'], user_api='blas'):
        model = solve(fitter, spec, module, columns, config.diagnostics['analytic_rcond'])
    del fitter
    diagnostics = {'module': module, 'candidate_columns': columns,
                   'fit': model.pop('diagnostics')}
    np.savez(target / 'predictor.npz', **model,
             feature_names=np.asarray(spec['feature_names']),
             target_channels=np.arange(meta['nch']), candidate_columns=np.asarray(columns, dtype=np.int64))
    (target / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+'\n')
    fitted = int(model['fitted'].sum())
    ranks = model['rank'][model['fitted']]
    underdetermined = sum(row['underdetermined'] for row in diagnostics['fit'])
    rank_deficient = sum(row['rank'] < row['active_predictors'] for row in diagnostics['fit']
                         if row['status'] != 'unavailable')
    summary = {'channels': meta['nch'], 'fitted_channels': fitted,
               'unavailable_channels': meta['nch']-fitted,
               'underdetermined_channels': underdetermined,
               'rank_deficient_channels': rank_deficient}
    print(f'{module}: {fitted}/{meta["nch"]} fitted; rank '
          f'{int(ranks.min()) if len(ranks) else 0}..{int(ranks.max()) if len(ranks) else 0}; '
          f'{underdetermined} underdetermined, {rank_deficient} rank deficient', flush=True)
    return summary, {'ROOT': ROOT.gROOT.GetVersion(), 'numpy': np.__version__}
