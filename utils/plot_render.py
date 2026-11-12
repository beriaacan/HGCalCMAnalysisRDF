"""Array/histogram rendering corresponding to the active reference apply -p calls."""
from __future__ import annotations

import json
from pathlib import Path
from contextlib import nullcontext

from .analytic_artifacts import sha
from .paths import module_plot_directory
from .inference_config import method_directory
from .plot_output import publish
from .plot_statistics import channel_statistics
from .plot_style import (matrix_figure, spectrum_figure, cell_area_axes, mean_violins,
                         window_label, column_label)


def _ratio(numerator,denominator):
    import numpy as np
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),
                     where=np.isfinite(numerator)&np.isfinite(denominator)&(denominator>0))


def noise_comparison_figure(raw, corrected, area, font_size):
    """Port the reference NoiseModelCohIncRatio presentation using cached arrays."""
    import numpy as np
    import matplotlib.pyplot as plt

    area=np.asarray(area)
    channel=np.arange(len(area))
    connected=area>0
    x=channel[connected]
    unconnected=channel[~connected]
    sigma={label:{key:np.sqrt(np.maximum(np.diag(data[key]),0.))
                  for key in ('incoherent','coherent')}
           for label,data in (('meas.',raw),('corr.',corrected))}

    fig=plt.figure(figsize=(8.2,6.6))
    grid=fig.add_gridspec(3,1,height_ratios=[3,1,1],hspace=.10,top=.84)
    top=fig.add_subplot(grid[0])
    ratio=fig.add_subplot(grid[1],sharex=top)
    coh_inc=fig.add_subplot(grid[2],sharex=top)
    for label,marker,size in (('meas.','.',4),('corr.','x',3)):
        for key,colour in (('incoherent','tab:blue'),('coherent','tab:orange')):
            top.plot(x,sigma[label][key][connected],marker,color=colour,ms=size,
                     label=f'{key} ({label})')
    for ax in (top,ratio,coh_inc):
        ax.tick_params(axis='both',direction='in',top=True,bottom=True,
                       left=True,right=True,labelsize=font_size)
        ax.grid(axis='y',ls='--',alpha=.3)
        for boundary in range(0,len(area)+1,37):
            ax.axvline(boundary,color='black',linestyle='--',linewidth=1,alpha=.35)
        for c in unconnected:
            ax.axvspan(c-.5,c+.5,color='lightgrey',alpha=1.,linewidth=0,zorder=3,
                       label='unconnected channels' if ax is top and c==unconnected[0] else None)
    top.set_ylabel('Noise (ADC)',fontsize=font_size*4/3,loc='top',labelpad=12)
    top.set_ylim(0.,top.get_ylim()[1]*1.2)
    top.legend(loc='upper right',fontsize=font_size)
    # Use each saved fit's actual rank: iterative selection can choose different
    # numbers of sources for measured and corrected data, including zero.
    tags=[]
    for label,fit in (('Measured',raw),('Corrected',corrected)):
        rank=np.asarray(fit['loadings']).shape[1]
        sources=('no coherent noise sources selected' if rank==0 else
                 f'{rank} coherent noise source'+('' if rank==1 else 's'))
        tags.append(f'{label}: {sources} (nc = {rank}).')
    top.set_title('\n'.join(tags),loc='left',fontsize=font_size*5/6,pad=8)
    for key,colour in (('incoherent','tab:blue'),('coherent','tab:orange')):
        values=_ratio(sigma['corr.'][key],sigma['meas.'][key])
        ratio.plot(x,values[connected],'x',color=colour,ms=3)
    ratio.set_ylabel('corr. / meas.',fontsize=font_size*11/12,loc='center',labelpad=10)
    ratio.set_ylim(0.,1.1)
    for label,marker,size in (('meas.','.',4),('corr.','x',3)):
        values=_ratio(sigma[label]['coherent'],sigma[label]['incoherent'])
        coh_inc.plot(x,values[connected],marker,color='black',ms=size)
    coh_inc.set_xlabel('channel',fontsize=font_size*4/3,loc='right',labelpad=8)
    coh_inc.set_ylabel('coh. / inc.',fontsize=font_size*11/12,loc='center',labelpad=8)
    coh_inc.set_ylim(0.,max(coh_inc.get_ylim()[1],2.))
    plt.setp(top.get_xticklabels(),visible=False)
    plt.setp(ratio.get_xticklabels(),visible=False)
    return fig


def _render_variant(config,module,info,products,work,variant):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from .plot_fits import json_safe
    options=config.plotting;groups=set(options['plot_groups'])
    from .noise_fixed_reference import label
    variant_label=label(config,variant)
    if variant != 'independent':
        groups &= {'noise_model','cell_area'}
    data={method:{key:product.arrays() for key,product in row.items()} for method,row in products.items()}
    summaries={method:{key:product.record['summary'] for key,product in row.items()} for method,row in products.items()}
    colours={'uncorrected':'black','analytic':'tab:green','analytic_allinputs':'tab:blue','dnn':'tab:red'}
    nch=info['nch'];channel=np.arange(nch);window=options['plot_noise_window']
    area=np.asarray(info['area']);fractions=np.asarray(info['area_fraction'])

    def build(work):
        fit_records={}
        def save(fig,relative,*,tight=True,**save_options):
            target=work/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if tight: fig.tight_layout()
            for extension in options['plot_formats']: fig.savefig(str(target)+'.'+extension,**save_options)
            plt.close(fig)
        def matrix_plot(matrix,title,relative,correlation=False):
            limits=(-1,1) if correlation else options['plot_covariance_range']
            fig=matrix_figure(matrix,title,limits,'corr(i,j)' if correlation else 'cov(i,j)')
            save(fig,relative)
        def spectrum(method,name,log=False):
            fig=spectrum_figure(data[method]['histograms'],summaries[method]['spectrum_fits'],
                                data[method]['spectrum_fits'],method,name,log,options)
            save(fig,Path('distributions_1d')/method_directory(config,method)/(f'{name}_1d'+('_logy' if log else '')))

        with plt.rc_context({'font.size':options['plot_font_size']*5/6}):
            for method,row in data.items():
                if 'covariance' in groups:
                    c=row['covariance']['covariance'].copy();c[row['covariance']['counts']<2]=np.nan
                    finite=np.where(np.isfinite(c),c,0.);total=np.sum(finite**2)
                    fraction=1.-np.sum(np.diag(finite)**2)/total if total>0 else np.nan
                    title=f'({column_label(method).replace("all","*",1)}) - f_corr = {fraction:.3g}'
                    matrix_plot(c,'Covariance '+title,Path('covcorr')/method_directory(config,method)/'Covariance')
                    sigma=np.sqrt(np.maximum(np.diag(c),0.));corr=_ratio(c,sigma[:,None]*sigma[None,:])
                    matrix_plot(np.clip(corr,-1,1),'Correlation '+title,Path('covcorr')/method_directory(config,method)/'Correlation',True)
                if 'noise_model' in groups:
                    fit=row['noise_model'];status=summaries[method]['noise_model'];suffix=variant_label
                    note='' if status['success'] else ' (fit not converged/unavailable)'
                    fit_title=(f'{variant_label}: {status["rank"]} coherent components'
                               + (f' | MSE = {status["mse"]:.3g}' if status.get('mse') is not None else '')
                               + (f' | coherent fraction = {status["coherent_fraction"]:.3g}' if status.get('coherent_fraction') is not None else ''))
                    for key,label in [('model','Fitted covariance'),('coherent','Coherent covariance'),('incoherent','Incoherent covariance'),('residual','Fit residual')]:
                        matrix_plot(fit[key],label+f' ({fit_title})'+note,Path('noise_model')/method_directory(config,method)/f'{key}_{suffix}')
                    fig,ax=plt.subplots(figsize=(8.2,4.2))
                    for values,label in [(np.diag(row['covariance']['covariance'])[:nch],'data diag'),(np.diag(fit['model']),'model diag'),
                                          (np.diag(fit['incoherent']),'incoherent var'),(np.diag(fit['coherent']),'coherent var')]: ax.plot(channel,values,label=label,lw=1.5)
                    ax.set_xlabel('channel');ax.set_ylabel('variance');ax.grid(ls='--',alpha=.3)
                    for pos in range(0,nch+1,37):ax.axvline(pos,color='black',ls='--',lw=1,alpha=.5)
                    ax.legend(fontsize='small',ncol=2)
                    ax.set_title(variant_label+note)
                    save(fig,Path('noise_model')/method_directory(config,method)/f'channel_variances_{suffix}')
                    fig,ax=plt.subplots(figsize=(8.2,4.2))
                    for j in range(fit['loadings'].shape[1]): ax.plot(channel,fit['loadings'][:,j],label=f'source {j+1}',lw=1.5)
                    ax.set_xlabel('channel');ax.set_ylabel('loading');ax.grid(ls='--',alpha=.3)
                    for pos in range(0,nch+1,37):ax.axvline(pos,color='black',ls='--',lw=1,alpha=.5)
                    ax.axhline(0,color='black',ls='--',lw=1,alpha=.5)
                    if fit['loadings'].shape[1]: ax.legend(fontsize='small')
                    else: ax.text(.5,.5,'nc=0: no coherent component',ha='center',transform=ax.transAxes)
                    ax.set_title(variant_label+note)
                    save(fig,Path('noise_model')/method_directory(config,method)/f'loadings_{suffix}')
                    if method!='uncorrected' and 'uncorrected' in data:
                        fig=noise_comparison_figure(data['uncorrected']['noise_model'],fit,area,options['plot_font_size'])
                        fig.suptitle(variant_label,fontsize=options['plot_font_size'],y=.99)
                        save(fig,Path('noise_model')/method_directory(config,method)/f'coherent_incoherent_comparison_{suffix}',
                             tight=False,bbox_inches='tight',pad_inches=.05)
                if 'spectra' in groups:
                    for name in (['gaussfit','sum']+(['mipfit'] if options['plot_mip_fit'] else [])):
                        for log in (False,True): spectrum(method,name,log)
                if 'channel_summaries' in groups:
                    base=data.get('uncorrected',row)['histograms']
                    for mode in ('full','window'):
                        mean,sigma,_=channel_statistics(row['histograms'],mode);bmean,bsigma,_=channel_statistics(base,mode)
                        suffix=' '+window_label(window) if mode=='window' else ''
                        for metric,values,baseline,title in [('mean',mean,bmean,'Mean'),('sigma',sigma,bsigma,'RMS')]:
                            fig,ax=plt.subplots(figsize=(8,5))
                            ax.plot(channel,baseline,'-',color='gray',label='Measured')
                            ax.plot(channel,values,'-',color='red',label='Corrected')
                            ax.set_xlabel('Channel index');ax.set_ylabel(title+suffix);ax.grid(True,alpha=.3);ax.legend()
                            save(fig,Path('summaries_1d')/method_directory(config,method)/f'{metric}_{mode}_vs_channel')
                            fig,ax=plt.subplots(figsize=(8,5));edges=np.linspace(*options[f'plot_{metric}_summary_range'],options[f'plot_{metric}_summary_bins']+1)
                            for v,label,colour in [(values,'Corrected','red'),(baseline,'Measured','gray')]:
                                counts,_=np.histogram(v[np.isfinite(v)],bins=edges)
                                ax.bar(edges[:-1],counts,width=np.diff(edges),align='edge',alpha=.6,color=colour,edgecolor=None,label=label)
                            ax.set_xlabel(column_label(method)+' '+('mean' if metric=='mean' else 'RMS')+suffix)
                            ax.set_ylabel('Number of channels');ax.set_ylim(bottom=0.);ax.grid(True,alpha=.3);ax.legend()
                            save(fig,Path('summaries_1d')/method_directory(config,method)/f'{metric}_{mode}_distribution')
                        ratio=_ratio(sigma,bsigma)
                        ratio_label='σ (corr.)'+suffix+' / σ (meas.)'+suffix
                        fig,ax=plt.subplots(figsize=(8,5));ax.plot(channel,ratio,'-',color='red')
                        ax.set_xlabel('Channel index');ax.set_ylabel(ratio_label);ax.grid(True,alpha=.3)
                        save(fig,Path('summaries_1d')/method_directory(config,method)/f'ratio_sigma_{mode}_vs_channel')
                        fig,ax=plt.subplots(figsize=(8,5));edges=np.linspace(*options['plot_ratio_summary_range'],options['plot_ratio_summary_bins']+1)
                        finite=ratio[np.isfinite(ratio)];counts,_=np.histogram(finite,bins=edges)
                        ax.step(.5*(edges[:-1]+edges[1:]),counts,where='mid',color='red')
                        if len(finite):
                            ax.axvline(finite.mean(),color='k',ls='-',label=f'mean = {finite.mean():.3f}, rms = {np.sqrt(np.mean(finite**2)):.3f}')
                            ax.legend()
                        ax.set_xlabel(ratio_label);ax.set_ylabel('Number of channels');ax.set_ylim(bottom=0.);ax.grid(True,alpha=.3)
                        save(fig,Path('summaries_1d')/method_directory(config,method)/f'ratio_sigma_{mode}_distribution')
            if 'cell_area' in groups:
                import mplhep as mh
                from matplotlib.lines import Line2D
                for key in (('coherent','incoherent') if variant != 'independent' else
                            ('window_sigma','window_mean','coherent','incoherent')):
                    context=plt.rc_context(mh.style.CMS) if key!='window_mean' else nullcontext()
                    with context:
                        fig,ax=plt.subplots(figsize=(8,5.5));prepared=[]
                        for method,row in data.items():
                            if key.startswith('window'):
                                mean,sigma,n=channel_statistics(row['histograms'],'window');values=sigma if key.endswith('sigma') else mean
                            else: values=np.sqrt(np.maximum(np.diag(row['noise_model'][key]),0.));n=np.ones(nch)
                            valid=(area>0)&np.isfinite(values)&(n>0);x=fractions[valid];y=values[valid];unique=np.unique(x)
                            grouped=[y[x==a] for a in unique];counts=np.asarray([len(g) for g in grouped])
                            prepared.append((method,x,y,unique,grouped,counts))
                        max_count=max((max(p[-1],default=0) for p in prepared),default=0)
                        scale=options['plot_font_size']/12.
                        for method,x,y,unique,grouped,counts in prepared:
                            if not len(unique):continue
                            means=np.asarray([g.mean() for g in grouped]);errors=np.asarray([g.std()/np.sqrt(len(g)) for g in grouped])
                            if key=='window_mean':mean_violins(ax,x,grouped,colours[method],counts,max_count)
                            label={'uncorrected':'No correction for CM','analytic':'CM-only linear correction','analytic_allinputs':'Linear correction for CM','dnn':'DNN correction for CM'}[method]
                            ax.errorbar(unique,means,yerr=errors,fmt='o',linewidth=1.6,markersize=5,capsize=4,color=colours[method],label=label)
                            if key!='window_mean' and len(unique)>=3:
                                try:
                                    p,_=curve_fit(lambda t,a,b,c:a*t*t+b*t+c,x,y,p0=(1.,1.,0.),maxfev=options['plot_spectrum_maxfev'])
                                    xx=np.linspace(x.min(),x.max(),200);ax.plot(xx,p[0]*xx*xx+p[1]*xx+p[2],'--',color=colours[method],lw=1.6,alpha=.9)
                                    fit_records[f'{key}/{method}']={'quadratic':p.tolist()}
                                except (ValueError,RuntimeError) as exc: fit_records[f'{key}/{method}']={'error':str(exc)}
                        cell_area_axes(fig,ax,info,config,key)
                        handles,legend_labels=ax.get_legend_handles_labels()
                        if key!='window_mean':
                            handles.append(Line2D([0],[0],ls='--',color='0.5',lw=1.6,alpha=.9));legend_labels.append('Quadratic fit')
                        ax.legend(handles,legend_labels,frameon=False,fontsize=(13 if key=='window_mean' else 15)*scale,
                                  loc='best' if key=='window_mean' else 'upper right')
                        if key in ('coherent','incoherent'): ax.set_title(variant_label)
                        stem=key+'_'+variant_label if key in ('coherent','incoherent') else key
                        save(fig,Path('summaries_compare')/'noise_vs_cell_area'/stem,tight=False)
        selections={}
        if any('noise_model' in row for row in data.values()):
            from .noise_selection import unfitted_offdiag_rmse, offdiag_mask
            original=data.get('uncorrected',{}).get('covariance')
            reference=(unfitted_offdiag_rmse(original['covariance'][:nch,:nch],original['counts'][:nch,:nch],fractions>0)
                       if original is not None else None)
            for method,row in data.items():
                if 'noise_model' not in row: continue
                status=summaries[method]['noise_model']
                mask=offdiag_mask(row['covariance']['covariance'][:nch,:nch],row['covariance']['counts'][:nch,:nch],fractions>0)
                rmse=float(np.sqrt(np.mean(row['noise_model']['residual'][mask]**2)))
                selections[method]={'rank':status['rank'],'offdiag_rmse_adc2':rmse,
                    'original_uncorrected_nc0_rmse_adc2':reference,
                    'unexplained_squared_offdiag_fraction_original':
                        (rmse/reference)**2 if reference is not None and reference>0 else None,
                    'fit_variant':variant_label,'fit':products[method]['noise_model'].token,
                    'reference_fit':status.get('reference_fit')}
            target=work/f'noise_model/fit_summary_{variant_label}.json';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(json_safe({'module':module,'methods':selections,
                'fraction_definition':'(RMSE(method,nc)/RMSE(uncorrected,0))^2; not total noise variance'}),indent=2,allow_nan=False)+'\n')
        (work/f'render_summary_{variant_label}.json').write_text(json.dumps(json_safe({'cell_area_fits':fit_records,'groups':sorted(groups),
            'noise_fits':selections}),indent=2)+'\n')
        return {'groups':sorted(groups),'methods':list(products),'window':window,
                'noise_ranks':{method:row['rank'] for method,row in selections.items()}}
    return build(work)


def render_module(config,module,info,products):
    """One publication containing both labelled fit variants and shared data plots."""
    from .noise_fixed_reference import fit_key, label
    variants=config.plotting['plot_noise_fit_variants']
    definition={'module':module,'products':{m:{k:p.token for k,p in row.items()} for m,row in products.items()},
        'settings':config.plotting,
        'geometry':{'area':info['area'],'fraction':info['area_fraction']},
        'renderer_sha256':sha(Path(__file__)), 'style_sha256':sha(Path(__file__).with_name('plot_style.py'))}
    def build(work):
        summaries={}
        has_fits=any('noise_model' in row for row in products.values())
        for variant in (variants if has_fits else ['independent']):
            selected={m:{k:p for k,p in row.items() if not k.startswith('noise_model')} for m,row in products.items()}
            if has_fits:
                for method,row in selected.items(): row['noise_model']=products[method][fit_key(variant)]
            summaries[label(config,variant)]=_render_variant(config,module,info,selected,work,variant)
        (work/'render_summary.json').write_text(json.dumps({'noise_fit_variants':summaries},indent=2)+'\n')
        return {'noise_fit_variants':summaries}
    result=publish(module_plot_directory(config,module),definition,build)
    print(f'Plots published: {result.path}',flush=True)
    return result
