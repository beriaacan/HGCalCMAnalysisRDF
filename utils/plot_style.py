"""Array-only presentation ported from the reference plotting functions.

Keep this module independent of event readers and fitted-product definitions.
The reference checkout is never imported at runtime.
"""
from __future__ import annotations

import numpy as np


def window_label(window):
    return '['+', '.join(f'{v:g}' for v in window)+']'


def column_label(method):
    suffix = {'uncorrected': '', 'analytic': '_resid_analytic_k0', 'analytic_allinputs': '_resid_analytic_allinputs', 'dnn': '_resid_dnn'}
    return 'adc_chall_pedsub'+suffix[method]


def pooled_moments(hist, mode):
    """Pool the saved exact channel moments without reconstructing events."""
    n, mean, m2 = (np.asarray(hist[f'{mode}_{key}']) for key in ('n', 'mean', 'm2'))
    valid = (n > 0) & np.isfinite(mean) & np.isfinite(m2)
    total = n[valid].sum()
    if total == 0:
        return np.nan, np.nan
    center = float(np.sum(n[valid]*mean[valid])/total)
    variance = np.sum(m2[valid]+n[valid]*(mean[valid]-center)**2)/total
    return center, float(np.sqrt(max(variance, 0.)))


def matrix_figure(matrix, title, limits, ztitle):
    """Draw genuine vector cells at the reference channel coordinates."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = np.arange(matrix.shape[0]+1)-.5
    image = ax.pcolormesh(edges, edges, np.ma.masked_invalid(matrix), shading='flat',
                          cmap='coolwarm', vmin=limits[0], vmax=limits[1],
                          rasterized=False, edgecolors='none', antialiased=False)
    for position in range(37, matrix.shape[0], 37):
        ax.axvline(position-.5, color='black', linestyle='--', linewidth=.7)
        ax.axhline(position-.5, color='black', linestyle='--', linewidth=.7)
    ticks = np.arange(0, matrix.shape[0], 37)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_title(title); ax.set_xlabel('channel i'); ax.set_ylabel('channel j')
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(ztitle)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    return fig


def spectrum_figure(hist, fits, curves, method, name, log, options):
    import matplotlib.pyplot as plt
    from .plot_fits import gaussian
    key = 'mip' if name == 'mipfit' else 'sum' if name == 'sum' else 'adc'
    counts, edges = hist[key], hist[key+'_edges']
    centers = .5*(edges[:-1]+edges[1:])
    fig, ax = plt.subplots(figsize=(8, 5.5 if name == 'mipfit' else 5))
    ax.step(centers, counts, where='mid', color='0.3' if name == 'mipfit' else 'gray',
            label='data' if name == 'mipfit' else None)
    ax.set_xlabel('adc_sum_pedsub'+column_label(method).removeprefix('adc_chall_pedsub')
                  if key == 'sum' else column_label(method))
    ax.set_ylabel('Entries' if name == 'mipfit' else 'Number of entries')
    if key == 'adc':
        for mode, linestyle in [('full', '-'), ('window', '--')]:
            mean, rms = pooled_moments(hist, mode)
            if np.isfinite(mean):
                label = 'mean' if mode == 'full' else 'mean '+window_label(options['plot_noise_window'])
                rlabel = 'rms' if mode == 'full' else 'rms '+window_label(options['plot_noise_window'])
                ax.axvline(mean, color='k', ls=linestyle,
                           label=f'{label} = {mean:.3f}, {rlabel} = {rms:.3f}')
    if name == 'gaussfit':
        result = fits['gaussian']
        if result['success']:
            p = result['parameters']; x = np.linspace(edges[0], edges[-1], 1000)
            ax.plot(x, gaussian(x, *p), label=f'gauss: A={p[0]:.1f}, mu={p[1]:.3f}, sigma={p[2]:.3f}')
        else:
            ax.text(.98, .97, 'Gaussian fit unavailable: '+result['message'],
                    ha='right', va='top', transform=ax.transAxes, fontsize='small')
    if name == 'mipfit':
        result = fits['mip']
        if result['success']:
            x = curves['x']
            ax.plot(x, sum(curves[k] for k in ('pedestal', 'single_mip', 'double_mip')), color='red', label='total fit')
            for component, label in [('pedestal', 'pedestal'), ('single_mip', '1 MIP'), ('double_mip', '2 MIP')]:
                ax.plot(x, curves[component], '--', label=label)
            p, e = result['parameters'], result['errors']
            uncertainty = lambda value: f'{value:.2g}' if value is not None and np.isfinite(value) else 'n/a'
            chi2 = result['chi2_ndof']
            note = (f'mu_p = {p[1]:.3g} +/- {uncertainty(e[1])}\n'
                    f'sigma_p = {p[2]:.3g} +/- {uncertainty(e[2])}\n'
                    f'Landau location = {p[4]:.3g} +/- {uncertainty(e[4])}\n'
                    f'Landau width c_L = {p[5]:.3g} +/- {uncertainty(e[5])}\n'
                    f'1-MIP peak = {result["one_mip_peak"]:.3g}\n'
                    + (f'chi2/ndof = {chi2:.2f}' if chi2 is not None else 'chi2/ndof = n/a'))
            if not result['fit_quality']['valid']:
                note += '\nFIT QUALITY WARNING'
        else:
            note = 'MIP fit unavailable: '+result['message']
        ax.text(.98, .97, note, ha='right', va='top', transform=ax.transAxes, fontsize=9,
                bbox={'facecolor':'white', 'alpha':.85})
    if log and np.any(counts > 0):
        ax.set_yscale('log')
        ax.set_ylim(bottom=.5 if name == 'mipfit' else 1., **({} if name == 'mipfit' else {'top':1.e7}))
    elif not log:
        ax.set_ylim(bottom=0.)
    ax.grid(ls='--' if name == 'mipfit' else '-', alpha=.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize='small' if name == 'mipfit' else None)
    return fig


def cell_area_axes(fig, ax, info, config, key):
    """Reference cell-area labels, CMS annotation and method-independent margins."""
    import mplhep as mh
    scale = config.plotting['plot_font_size']/12.
    mean_plot = key == 'window_mean'
    titles = {'window_sigma':'Mean noise (ADC units)', 'window_mean':'Mean ADC '+window_label(config.plotting['plot_noise_window']),
              'coherent':'Mean coherent noise (ADC units)', 'incoherent':'Mean incoherent noise (ADC units)'}
    ax.set_xlabel('Cell area / full cell area' if mean_plot else 'Cell area / standard cell area',
                  fontsize=(16 if mean_plot else 20)*scale, loc='right', labelpad=8)
    ax.set_ylabel(titles[key], fontsize=(16 if mean_plot else 20)*scale, loc='top', labelpad=10)
    ax.tick_params(axis='both', which='both', labelsize=(13 if mean_plot else 16)*scale)
    if mean_plot:
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=6)
    else:
        mh.cms.label(llabel='Preliminary', rlabel='', ax=ax, data=True, loc=2, fontsize=22*scale)
    area, fractions = np.asarray(info['area']), np.asarray(info['area_fraction'])
    standard = area[np.isfinite(area) & np.isclose(fractions, 1.)]
    if len(standard):
        ax.text(.05, .75, f'Standard cell area: {standard[0]:.2f} mm$^2$', transform=ax.transAxes,
                fontsize=(13 if mean_plot else 15)*scale, ha='left', va='top', color='0.0')
    runs = [s.run for s in config.target.dataset.sources]
    # Keep the reference annotation position without guessing beam species.
    run_type = ('Pedestal run' if runs == [config.target.pedestal.run] else
                f'Run {runs[0]}' if len(runs) == 1 else 'Multiple runs')
    ax.text(.05, .68, run_type, transform=ax.transAxes, fontsize=(13 if mean_plot else 15)*scale,
            ha='left', va='top', color='0.0')
    if not mean_plot:
        ax.set_ylim(*config.plotting['plot_noise_area_range'])
    fig.subplots_adjust(left=.10 if mean_plot else .12 if config.plotting['plot_noise_area_range'][1] < 5 else .08,
                        right=.98, bottom=.14, top=.98)


def mean_violins(ax, fractions, grouped, color, group_counts, max_count):
    """Reference mean-ADC channel distributions, with degenerate groups omitted."""
    unique = np.unique(fractions)
    spacing = np.min(np.diff(unique)) if len(unique) > 1 else .2
    widths = 5.*spacing*np.sqrt(group_counts/max(max_count, 1))
    usable = [(x, y, w) for x, y, w in zip(unique, grouped, widths) if len(y)>1 and np.ptp(y)>0]
    if not usable:
        return
    violin = ax.violinplot([y for _, y, _ in usable], positions=[x for x, _, _ in usable],
                          widths=[w for _, _, w in usable], bw_method=.1,
                          showmeans=False, showmedians=False, showextrema=False, points=100)
    for body in violin['bodies']:
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.18); body.set_linewidth(1.8)
