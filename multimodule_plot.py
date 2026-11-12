"""Run-level summaries with checked, reusable covariance/fit prerequisites."""
from __future__ import annotations

from pathlib import Path

from utils.config import AnalysisConfig
from utils.paths import resolve_paths, application_plot_directory


def destination(config):
    return application_plot_directory(config, Path(resolve_paths(config)['target']['plots'])/'multimodule'/
            f'pedestals_from_Run{config.target.pedestal.run}'/config.model_dir.name/config.target.selection)


def describe(config):
    return {'destination': str(destination(config)), 'methods': config.methods,
            'settings': config.multimodule, 'fit_variants':config.plotting['plot_noise_fit_variants'],
            'sources': 'audited covariance/noise-fit caches; calculate missing covariances from saved application data and missing fits',
            'individual_module_figures': False,
            'geometry': 'central wafer templates, with Nano Runs / configured locator cross-check',
            'layout': 'Cassette layers' if config.target.campaign == 'Cassette' else 'layer-ordered June stack',
            'publication': 'fixed local figures overwritten; source/product hashes in manifest and values in summary.json'}


def run(config: AnalysisConfig) -> None:
    import plots
    from utils.analytic_artifacts import sha
    from utils.multimodule_statistics import noise_products
    from threadpoolctl import threadpool_limits
    from utils.multimodule_geometry import central_polygons, positions, detector_geometries
    from utils.multimodule_render import render
    from utils.plot_output import publish
    groups = config.multimodule['multimodule_plot_groups']
    if 'uncorrected' not in config.methods:
        raise ValueError('multimodule_plot requires uncorrected in methods as the comparison baseline')
    if 'performance' in groups and not any(m != 'uncorrected' for m in config.methods):
        raise ValueError('Multimodule performance requires a corrected method alongside uncorrected')
    print('Checking saved covariance/noise-fit provenance for multimodule summaries', flush=True)
    modules = plots.resolve_inputs(config)
    if len(modules) < 2:
        raise ValueError('multimodule_plot requires at least two selected application modules')
    with threadpool_limits(limits=config.resources['cpus'], user_api='blas'):
        products = noise_products(config, modules)
    polygons, geometries, geometry_sources = {}, [], {}
    if set(groups) & {'detector_maps', 'canonical_maps'}:
        polygons, geometry_sources = central_polygons(config, modules)
    if 'detector_maps' in groups:
        placement, placement_sources = positions(config, modules)
        geometry_sources['placement'] = placement_sources
        geometries = detector_geometries(config, modules, polygons, placement)
    root = Path(__file__).resolve().parent
    definition = {'schema': 'hgcal_cm_multimodule_v1', 'campaign': config.target.campaign,
        'runs': [s.run for s in config.target.dataset.sources], 'methods': list(config.methods),
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in config.multimodule.items()},
        'formats': config.plotting['plot_formats'], 'font_size': config.plotting['plot_font_size'],
        'products': {v:{n:{m:{'covariance':row['covariance'].token,'fit':row['fit'].token}
            for m,row in methods.items()} for n,methods in variant.items()} for v,variant in products.items()},
        'geometry': geometry_sources,
        'connected_channels': {n: modules[n]['area_fraction'] for n in modules},
        'definitions': {'performance': 'median of per-channel RMS ratios, separately on each axis; connected channels only',
                        'canonical': 'per-module component median subtracted, divided by module median total RMS; median across modules of same wafer type',
                        'closure': 'measured RMS / fitted total RMS - 1',
                        'ratios': 'corrected / uncorrected fitted noise RMS; invalid or zero denominators unavailable'},
        'code_sha256': {name: sha(root/name) for name in ('multimodule_plot.py',
            'utils/multimodule_statistics.py', 'utils/multimodule_geometry.py', 'utils/multimodule_render.py',
            'utils/multimodule_config.py','utils/noise_fixed_reference.py')}}
    def build(work):
        import json
        summaries={v:render(config,modules,rows,geometries,polygons,work,fit_variant=v)
                   for v,rows in products.items()}
        (work/'summary.json').write_text(json.dumps({'fit_variants':summaries},indent=2,allow_nan=False)+'\n')
        return {'fit_variants':summaries}
    publish(destination(config),definition,build)
    print(f'Multimodule plots ready: {destination(config)}', flush=True)
