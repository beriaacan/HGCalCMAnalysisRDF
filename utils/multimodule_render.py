"""Array-only ports of the reference run-level noise summaries and maps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .inference_config import method_directory
from .multimodule_statistics import performance, ratio, scaled_median
from .plot_fits import json_safe

COMPONENTS = [('total', r'Total noise $\sigma_{\mathrm{tot}}$'),
              ('incoherent', r'Incoherent noise $\sigma_{\mathrm{inc}}$'),
              ('coherent', r'Coherent noise $\sigma_{\mathrm{coh}}$')]
COLORS = ['#3F90DA', '#FFA90E', '#BD1F01', '#94A4A2', '#832DB6', '#A96B59',
          '#E76300', '#B9AC70', '#717581', '#92DADD', '#1845FB']
NOISE_COLORS = ('#fff9e6', '#f7dc8a', '#e7a64a', '#c7683f', '#81384a', '#292331')
RELATIVE_COLORS = ('#426985', '#9eb9c5', '#f4f1ea', '#d7a06f', '#914c3e')
LABELS = {'uncorrected': 'Uncorrected', 'analytic': 'CM-only linear correction', 'analytic_allinputs': 'Analytic correction', 'dnn': 'DNN correction'}


def reference_family_counts(config, layers):
    """Inventory of reference families plus the shared June ratio/closure extension.

    Counts are figures, before multiplying by requested file formats. A layer
    ordered June stack counts as one geometry. Canonical figures each contain
    all selected wafer types, with an additional three-component summary per method.
    Ratio and closure families apply to both campaigns.
    """
    groups = config.multimodule['multimodule_plot_groups']
    methods = len(config.methods)
    corrected = sum(method != 'uncorrected' for method in config.methods)
    counts = {}
    if 'performance' in groups: counts['performance'] = corrected
    if 'detector_maps' in groups:
        counts['detector_maps'] = 1+(layers if layers > 1 else 0)+3*methods*layers
        counts['closure_maps'] = 1+methods*layers
        counts['ratio_maps'] = 4*corrected*layers
    if 'canonical_maps' in groups: counts['canonical_maps'] = 4*methods
    return counts


def histogram_with_flow(values, edges):
    finite = np.asarray(values)[np.isfinite(values)]
    clipped = np.clip(finite, np.nextafter(edges[0], edges[-1]), np.nextafter(edges[-1], edges[0]))
    return np.histogram(clipped, edges)[0], {'underflow': int((finite < edges[0]).sum()),
                                          'overflow': int((finite > edges[-1]).sum())}


def performance_figure(data, method, options):
    import matplotlib.pyplot as plt
    import mplhep as mh
    from matplotlib.lines import Line2D
    limits = options['multimodule_performance_range']
    edges = np.linspace(*limits, options['multimodule_performance_bins']+1)
    x, xflow = histogram_with_flow(data['channel_coherent_rejection'], edges)
    y, yflow = histogram_with_flow(data['channel_incoherent_retention'], edges)
    fig = plt.figure(figsize=(9.3, 7.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(4.5, 1.15), height_ratios=(1.15, 4.5),
                           left=.105, right=.985, bottom=.125, top=.98, wspace=.04, hspace=.04)
    top = fig.add_subplot(grid[0, 0]); main = fig.add_subplot(grid[1, 0], sharex=top)
    right = fig.add_subplot(grid[1, 1], sharey=main)
    top.stairs(x, edges, fill=True, color='.65', linewidth=0); top.set_ylabel('Channels', loc='top')
    top.tick_params(axis='x', labelbottom=False)
    right.stairs(y, edges, orientation='horizontal', fill=True, color='.65', linewidth=0)
    right.set_xlabel('Channels', loc='right'); right.tick_params(axis='y', labelleft=False)
    families = list(dict.fromkeys(row['family'] for row in data['modules']))
    markers = {family: ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>'][i % 9] for i, family in enumerate(families)}
    grouped = len(data['modules']) > len(COLORS)
    handles, labels = [], []
    for i, row in enumerate(data['modules']):
        color = '.35' if grouped else COLORS[i]
        points = row['erx_points']; valid = np.isfinite(points).all(axis=1)
        main.scatter(points[valid, 0], points[valid, 1], s=30, marker=markers[row['family']],
                     color=color, alpha=.2 if grouped else .3, linewidths=0, zorder=2)
        handle = main.scatter(*row['point'], s=125, marker=markers[row['family']], color=color,
                              edgecolor='black', linewidth=.8, zorder=4)
        if not grouped: handles.append(handle); labels.append(row['module'])
    if grouped:
        for family in families:
            handles.append(Line2D([], [], linestyle='none', marker=markers[family], markersize=9,
                                  markerfacecolor='.35', markeredgecolor='black', markeredgewidth=.8))
            labels.append(family)
    handles.append(main.scatter(1., 1., s=180, marker='*', facecolor='none', edgecolor='black', linewidth=1.2))
    labels.append('Ideal')
    main.axvline(1., color='.35', ls='--', lw=1.); main.axhline(1., color='.35', ls='--', lw=1.)
    main.set_xlim(limits); main.set_ylim(limits)
    main.set_xlabel(r'Coherent-noise rejection $1-\sigma_{\mathrm{coh}}^{\mathrm{corr}}/\sigma_{\mathrm{coh}}^{\mathrm{meas}}$', loc='right')
    main.set_ylabel(r'Incoherent-noise retention $\sigma_{\mathrm{inc}}^{\mathrm{corr}}/\sigma_{\mathrm{inc}}^{\mathrm{meas}}$', loc='top')
    for ax in (top, main, right):
        ax.minorticks_on(); ax.tick_params(which='both', direction='in', top=True, right=True)
    mh.cms.label(llabel='Preliminary', supp=LABELS[method], rlabel='', ax=main, data=True, loc=1,
                 fontsize=(29, 22, 17, 14), fontweight=('bold', 'normal', 'normal', 'normal'))
    main.legend(handles, labels, loc='lower left', bbox_to_anchor=(.015, .035), ncol=2,
                frameon=False, fontsize=11, handletextpad=.45, columnspacing=.9)
    return fig, {'coherent_histogram': xflow, 'incoherent_histogram': yflow,
                 'points_outside_display': [r['module'] for r in data['modules']
                     if not all(limits[0] <= p <= limits[1] for p in r['point'])]}


def map_panel(ax, geometry, values, cmap, norm, title, method):
    from matplotlib.collections import PolyCollection, LineCollection
    collection = PolyCollection(geometry['polygons'], array=np.ma.masked_invalid(values),
        cmap=cmap, norm=norm, edgecolors=(.35, .35, .35, .32), linewidths=.055, rasterized=False)
    ax.add_collection(collection)
    if geometry['outlines']:
        ax.add_collection(LineCollection(geometry['outlines'], colors='black', linewidths=.38))
    ax.set_xlim(geometry['xlim']); ax.set_ylim(geometry['ylim']); ax.set_aspect('equal')
    ax.set_title(title); ax.tick_params(which='both', direction='in', top=True, right=True)
    if geometry['coordinates']:
        ax.set_xlabel('$x$ [cm]', loc='right'); ax.set_ylabel('$y$ [cm]', loc='top')
    else:
        ax.set_xticks([]); ax.set_yticks([])
    for x, y, label in geometry['annotations']:
        ax.text(x, y, label, ha='center', va='bottom', fontsize=7.5)
    # Layer stacks leave a free bottom-right slot, matching the reference.
    x, y = geometry.get('label_position', (.045, .955))
    ax.text(x, y, 'CMS', transform=ax.transAxes, ha='left', va='top', fontsize=19, fontweight='bold')
    ax.text(x, y-.080, 'Preliminary', transform=ax.transAxes, ha='left', va='top', fontsize=13, fontstyle='italic')
    ax.text(x, y-.160, LABELS.get(method, method), transform=ax.transAxes, ha='left', va='top', fontsize=11)
    return collection


def render(config, modules, products, geometries, polygons, work, *, fit_variant):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    options = config.multimodule; groups = options['multimodule_plot_groups']
    from .noise_fixed_reference import label
    fit_label = label(config,fit_variant)
    selected_ranks = {name: {method: row['fit'].record['summary']['rank'] for method, row in methods.items()}
                      for name, methods in products.items()}
    summary = {'selected_ranks': selected_ranks, 'fit_variant':fit_label,'rank':config.diagnostics['n_coherent_noise_model'],
               'performance': {}, 'canonical': {}, 'display_ranges': {}, 'fit_status': {
        name: {m: row['fit'].record['summary'] for m, row in methods.items()} for name, methods in products.items()}}
    inventory = []

    def save(fig, path):
        if str(path) in inventory:
            raise ValueError(f'Duplicate multimodule figure path: {path}')
        inventory.append(str(path))
        target = work/path; target.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle(fit_label,fontsize=10,y=1.01)
        try:
            for extension in config.plotting['plot_formats']:
                fig.savefig(str(target)+'.'+extension, bbox_inches='tight', pad_inches=.02)
        finally:
            plt.close(fig)

    def grid(panels, ncols, cmap, norm, ztitle, path, extend):
        rows = int(np.ceil(len(panels)/ncols))
        fig, axes = plt.subplots(rows, ncols, figsize=(5*ncols, 4.15*rows+.75),
                                 squeeze=False, constrained_layout=True)
        for ax, (geometry, values, title, method) in zip(axes.flat, panels):
            collection = map_panel(ax, geometry, values, cmap, norm, title, method)
        for ax in list(axes.flat)[len(panels):]: ax.set_visible(False)
        cbar = fig.colorbar(collection, ax=list(axes.flat), fraction=.025 if ncols > 1 else .045,
                            pad=.018, aspect=38 if ncols > 1 else 30, extend=extend)
        cbar.set_label(ztitle)
        if cbar.solids is not None: cbar.solids.set_rasterized(False)
        save(fig, path)

    with plt.rc_context({'font.size': config.plotting['plot_font_size'], 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        if 'performance' in groups:
            for method in config.methods:
                if method == 'uncorrected': continue
                data = performance(modules, products, method, options['multimodule_min_coherent_sigma'])
                fig, flow = performance_figure(data, method, options)
                save(fig, Path('performance')/method_directory(config, method)/f'NoiseModelMultimodulePerformance_{fit_label}')
                summary['performance'][method] = dict(data, display=flow)

        cmap = LinearSegmentedColormap.from_list('noise', NOISE_COLORS); cmap.set_bad('white')
        relative = LinearSegmentedColormap.from_list('noise_relative', RELATIVE_COLORS); relative.set_bad('white')
        if 'detector_maps' in groups:
            values = {(i, m, c): np.asarray([products[n][m]['data'][c][ch] for n, ch in g['refs']])
                      for i, g in enumerate(geometries) for m in config.methods for c, _ in COMPONENTS+[('closure', '')]}
            lower, upper = options['multimodule_noise_range']
            norm = Normalize(lower, upper); summary['display_ranges']['noise'] = [lower, upper]
            panels = [(g, values[i, m, c], f'{g["detail"]}: {title}', m)
                      for i, g in enumerate(geometries) for m in config.methods for c, title in COMPONENTS]
            name = 'Cassette' if config.target.campaign == 'Cassette' else 'TestBeam'
            grid(panels, 3, cmap, norm, 'Noise [ADC counts]', Path('detector_maps')/f'NoiseModel{name}Maps_{fit_label}', 'both')
            for i, g in enumerate(geometries):
                layer = f'_layer{g["detail"].split()[-1]}' if config.target.campaign == 'Cassette' else ''
                if len(geometries) > 1:
                    grid([(g, values[i, m, c], title, m) for m in config.methods for c, title in COMPONENTS],
                         3, cmap, norm, 'Noise [ADC counts]', Path('detector_maps')/f'NoiseModel{name}Maps{layer}_{fit_label}', 'both')
                for m in config.methods:
                    for c, title in COMPONENTS:
                        grid([(g, values[i, m, c], title, m)], 1, cmap, norm, 'Noise [ADC counts]',
                             Path('detector_maps')/method_directory(config, m)/f'{c}{layer}_{fit_label}', 'both')
            lower, upper = options['multimodule_closure_range']
            summary['display_ranges']['closure'] = [lower, upper]
            norm_closure = TwoSlopeNorm(vmin=lower, vcenter=0., vmax=upper)
            panels = [(g, values[i, m, 'closure'], g['detail'], m)
                      for i, g in enumerate(geometries) for m in config.methods]
            grid(panels, len(config.methods), relative, norm_closure, r'Measured RMS / fitted RMS $-1$',
                 Path('closure_maps')/f'NoiseModel{name}ClosureMaps_{fit_label}', 'both')
            for m in config.methods:
                for i, g in enumerate(geometries):
                    grid([(g, values[i, m, 'closure'], g['detail'], m)], 1, relative, norm_closure,
                         r'Measured RMS / fitted RMS $-1$', Path('closure_maps')/method_directory(config, m)/f'{g["detail"].replace(" ", "")}_{fit_label}', 'both')
            for m in config.methods:
                if m == 'uncorrected': continue
                ratios = {(i, c): ratio(values[i, m, c], values[i, 'uncorrected', c])
                          for i in range(len(geometries)) for c, _ in COMPONENTS}
                lower, upper = options['multimodule_ratio_range']
                rnorm = TwoSlopeNorm(vmin=lower, vcenter=1., vmax=upper)
                summary['display_ranges'][m+'_ratio'] = [lower, upper]
                for i, g in enumerate(geometries):
                    # Reuse the three-component overview for each geometry:
                    # one June stack or each Cassette layer.
                    suffix = f'_layer{g["detail"].split()[-1]}' if len(geometries) > 1 else ''
                    grid([(g, ratios[i, c], title, m) for c, title in COMPONENTS], 3,
                         relative, rnorm, 'Corrected / uncorrected noise RMS',
                         Path('ratio_maps')/method_directory(config, m)/f'NoiseModel{name}RatioMaps{suffix}_{fit_label}', 'both')
                    for c, title in COMPONENTS:
                        grid([(g, ratios[i, c], title, m)], 1, relative, rnorm, 'Corrected / uncorrected noise RMS',
                             Path('ratio_maps')/method_directory(config, m)/f'{c}_{g["detail"].replace(" ", "")}_{fit_label}', 'both')

        if 'canonical_maps' in groups:
            from .multimodule_geometry import finish
            types = sorted({name[:4] for name in modules})
            canonical = []
            for kind in types:
                names = [n for n in modules if n[:4] == kind]
                connected = np.any([np.asarray(modules[n]['area_fraction']) > 0 for n in names], axis=0)
                channels = np.flatnonzero(connected)
                g = finish([polygons[kind][ch] for ch in channels], [], [], [], kind, False)
                g['available_modules'] = len(names)
                data = {}
                summary['canonical'][kind] = {}
                for m in config.methods:
                    summary['canonical'][kind][m] = {}
                    for c, _ in COMPONENTS:
                        value, counts, contributors = scaled_median(names, modules, products, m, c)
                        data[m, c] = value[channels]
                        summary['canonical'][kind][m][c] = {'values': value, 'cell_counts': counts,
                                                          'contributing_modules': contributors, 'available_modules': len(names)}
                canonical.append((g, data))
            lower, upper = options['multimodule_relative_range']
            norm = TwoSlopeNorm(vmin=lower, vcenter=0., vmax=upper)
            summary['display_ranges']['canonical'] = [lower, upper]
            ztitle = r'Median $(\sigma_c-\mathrm{median}_{ch}\,\sigma_c)/\mathrm{median}_{ch}\,\sigma_{tot}$'
            for m in config.methods:
                def canonical_title(g, component, title):
                    available = g['available_modules']
                    used = summary['canonical'][g['detail']][m][component]['contributing_modules']
                    count = 'N = 1; individual' if available == 1 else 'N = 2; limited' if available == 2 else (
                        f'N = {used}' if used == available else f'N = {used}/{available}')
                    return f'{g["detail"].replace("_", " ")} ({count}): {title}'
                # The summary reuses these exact arrays and the same shared
                # color scale as the individual component maps.
                grid([(g, d[m, c], canonical_title(g, c, title), m)
                      for g, d in canonical for c, title in COMPONENTS],
                     3, relative, norm, ztitle,
                     Path('canonical_maps')/method_directory(config, m)/f'summary_{fit_label}', 'both')
                for c, title in COMPONENTS:
                    grid([(g, d[m, c], canonical_title(g, c, title), m) for g, d in canonical],
                         min(3, len(canonical)), relative, norm, ztitle,
                         Path('canonical_maps')/method_directory(config, m)/f'{c}_{fit_label}', 'both')
    expected = reference_family_counts(config, len(geometries))
    actual = {family: sum(Path(path).parts[0] == family for path in inventory) for family in expected}
    if actual != expected:
        raise RuntimeError(f'Multimodule plot inventory differs from expected families: expected {expected}, produced {actual}')
    summary['plot_inventory'] = {'reference_family_counts': expected, 'produced_family_counts': actual,
                                 'figure_stems': inventory, 'formats': config.plotting['plot_formats']}
    (work/f'summary_{fit_label}.json').write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False)+'\n')
    return json_safe({'modules': list(modules), 'methods': list(config.methods),
                      'fit_variant':fit_label,'rank':config.diagnostics['n_coherent_noise_model'], 'selected_ranks': selected_ranks,
                     'groups': groups, 'display_ranges': summary['display_ranges'], 'plot_inventory': summary['plot_inventory'],
                     'nonconverged_fits': [f'{n}/{m}' for n in products for m in products[n]
                         if not products[n][m]['fit'].record['summary'].get('success', False)]})
