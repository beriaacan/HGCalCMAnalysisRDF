"""Reproduce the active legacy apply -p families from cached RDF statistics."""
from __future__ import annotations
from utils.prepared import prepared_manifest, module_signature, select_modules

import json
from pathlib import Path
from dataclasses import replace
from utils.inference_config import method_directory

from utils.analytic_artifacts import sha
from utils.paths import resolve_paths
from utils.plot_config import histogram_definition, histogram_label
from utils import plot_cache as cache
from utils.plot_fit_provenance import spectrum_code_hash
from utils.noise_fixed_reference import attach_fits, label as fit_label

ROOT_DIR=Path(__file__).resolve().parent
METHODS={'uncorrected','analytic','analytic_allinputs','dnn'}


def describe(config):
    paths=resolve_paths(config)['target']
    return {'runs':[s.run for s in config.target.dataset.sources], 'methods':config.methods,
            'groups':config.plotting['plot_groups'],'statistics':str(Path(paths['covariances']).parent/'statistics'),
            'covariances':paths['covariances'],'noise_fits':paths['noise_model_fits'],'plots':paths['plots'],
            'source_tracking':'per-file manifests with original UUID, prepared/prediction identity, selections and statistic definition',
            'reuse':'independent covariance, histogram and fit caches; figures redrawn and overwritten at fixed local paths',
            'noise_fit_variants':[fit_label(config, v) for v in config.plotting['plot_noise_fit_variants']],
            'noise_rank_selection':'paused; fixed configured rank for every stage',
            'plot_settings':config.plotting,'event_source':'configured application run; no training validation evaluation'}


def resolve_inputs(config):
    """Reuse framework preflight; never recreate prepared inputs or predictions."""
    import apply as application
    from utils.prepared import plan,resolve,cached_files
    if set(config.methods)-METHODS:
        raise NotImplementedError('plots supports uncorrected, analytic, analytic_allinputs and dnn')
    if config.target.selection not in {'full','trigtime'}:
        raise NotImplementedError('plots currently supports full or trigtime application selections')
    modules={}
    for job in plan(config):
        if 'apply' not in job.roles: continue
        prepared=resolve(config,job)
        prepared=select_modules(prepared, application._selected(config.target.modules,prepared['modules']))
        if cached_files(prepared) is None: raise FileNotFoundError('Run --steps prepare before plots')
        manifest=prepared_manifest(prepared)
        for name in application._selected(config.target.modules,prepared['modules']):
            info=modules.setdefault(name,{'nch':37*prepared['modules'][name]['identity'][2],
                                         'nerx':prepared['modules'][name]['identity'][2],'sources':[]})
            for source in prepared['files']:
                original=next(x for x in manifest['outputs'] if x['module']==name and x['source_file_index']==source['index'])
                info['sources'].append({'run':job.source.run,'source_file_index':source['index'],'source_uuid':source['uuid'],
                    'input_file':str(prepared['destination']/original['file']),'input_entries':source['entries'],
                    'input_size':original['size'],'input_mtime_ns':original['mtime_ns'],
                    'prepared_signature':module_signature(prepared,name), 'corrections':{}})
    for method in config.methods:
        if method=='uncorrected': continue
        if method=='dnn':
            from utils.dnn_application import resolve_application
            resolved=resolve_application(config)
        elif method=='analytic':
            from utils.cm_application import resolve_application
            resolved=resolve_application(config)
        else:
            resolved=application.resolve_application(replace(config,methods=('analytic_allinputs',)))
        path=resolved['destination']/'manifest.json'
        if not path.is_file(): raise FileNotFoundError(f'Run --steps apply before plotting {method}')
        completed=application.cached_application(resolved)
        stored=resolved['definition']
        for name,row in resolved['modules'].items():
            applied=completed['module_records'].get(name)
            if applied is None: raise FileNotFoundError(f'{method} application is incomplete for {name}')
            lookup={(r['run'],r['source_file_index']):r for r in row['inputs']}
            for source in modules[name]['sources']:
                item=lookup[(source['run'],source['source_file_index'])]
                output=next(x for x in applied['outputs'] if x['file']==item['output_file'])
                source['corrections'][method]={'path':str(resolved['destination']/name/output['file']),
                    'size':output['size'],'mtime_ns':output['mtime_ns'],'selected_events':output['selected_events'],
                    'predictor':row['definition']['predictor'], 'predictor_sha256':row['definition']['predictor_sha256'],
                    'application_code':stored['code_sha256']}
    areas=json.loads(config.dnn['cell_area_file'].read_text())
    for name,info in modules.items():
        geometry=areas.get(name[:4],{})
        if any(len(geometry.get(key,[]))!=info['nch'] for key in ('Area','SF')):
            raise ValueError(f'Central cell-area geometry length mismatch: {name}')
        info['area']=geometry['Area'];info['area_fraction']=geometry['SF']
        for source in info['sources']:
            manifest=json.loads((Path(source['input_file']).parent/'manifest.json').read_text())
            definition=manifest['definition']
            source['preparation_recipe']={k:v for k,v in definition.items() if k not in {'files','module_order','modules','run'}}
            source['module_identity']=definition['modules'][name]
    if not modules: raise ValueError('No plotting modules selected')
    return modules


def source_definition(config,module,source,kind,method):
    selection={'campaign':config.target.campaign,'selection':config.target.selection,
               'trigger_time_window':list(config.target.trigger_time_window),
               'adc_min':config.target.dataset.adcmin,'adc_max':config.target.dataset.adcmax}
    identity={key:source[key] for key in ('run','source_file_index','source_uuid','input_file','input_entries',
              'input_size','input_mtime_ns','preparation_recipe','module_identity')}
    if method in {'analytic','analytic_allinputs','dnn'}: identity['correction']=source['corrections'][method]
    return {'schema':'hgcal_cm_plot_statistics_v1','module':module,'kind':kind,'method':method,
            'source':identity,'selection':selection,
            'statistic':histogram_definition(config.plotting) if kind=='histograms' else
                        'pairwise finite, separately centered population covariance; raw per-eRx CM; save counts',
            'code_sha256':{name:sha(ROOT_DIR/name) for name in ('utils/plot_statistics.py','utils/plot_helpers.hxx')}}


def _numeric_product(directory,definition,arrays,summary):
    import numpy as np
    def build(work):
        np.savez_compressed(work/'data.npz',**arrays)
        return summary
    return cache.publish(directory,definition,build)


def contributions(config,module,info):
    from utils.plot_statistics import collect
    paths=resolve_paths(config)['target'];root=Path(paths['covariances']).parent/'statistics'/module
    want_cov=bool(set(config.plotting['plot_groups']) & {'covariance','noise_model','cell_area'})
    want_hist=bool(set(config.plotting['plot_groups']) & {'spectra','channel_summaries','cell_area'})
    wanted=([('cm',None)] if want_cov else [])
    wanted += [(kind,method) for method in config.methods for kind,yes in [('covariance',want_cov),('histograms',want_hist)] if yes]
    products={key:[] for key in wanted}
    for source in info['sources']:
        pending={};known={}
        for kind,method in wanted:
            directory=root/(method_directory(config,method) if method else Path('common'))/kind
            if kind=='histograms': directory/=histogram_label(config.plotting)
            directory/=f'Run{source["run"]}'
            directory/=Path(source['input_file']).stem
            definition=source_definition(config,module,source,kind,method)
            product=cache.matching(directory,definition,immutable=True)
            if product is None: pending[(kind,method)]=(directory,definition)
            else:
                print(f'Reusing {module} {method or "common CM"} {kind}: Run{source["run"]}/file{source["source_file_index"]}',flush=True)
                known[(kind,method)]=product
        if pending:
            print(f'Calculating {module} Run{source["run"]}/file{source["source_file_index"]}: {list(pending)}',flush=True)
            arrays={}
            base={key for key in pending if key[1] in {None,'uncorrected'}}
            if base: arrays.update(collect(config,source,info['nch'],info['nerx'],base))
            for method in config.methods:
                method_keys={key for key in pending if key[1]==method and method!='uncorrected'}
                if not method_keys: continue
                view=dict(source,correction=source['corrections'][method])
                translated={(kind,'analytic_allinputs') for kind,_ in method_keys}
                values=collect(config,view,info['nch'],info['nerx'],translated)
                arrays.update({(kind,method):values[(kind,'analytic_allinputs')] for kind,_ in method_keys})
            for key,(directory,definition) in pending.items():
                known[key]=_numeric_product(directory,definition,arrays[key],
                    {'events':int(arrays[key]['events']),'prepared_run_signature':source['prepared_signature']})
        for key in wanted: products[key].append(known[key])
    return products


def combine(config,module,key,products):
    from utils.plot_statistics import merge
    kind,method=key
    root=Path(resolve_paths(config)['target']['covariances']).parent/'statistics'/module/(method_directory(config,method) if method else Path('common'))/kind
    if kind=='histograms': root/=histogram_label(config.plotting)
    directory=root/'combined'
    definition={'kind':kind,'method':method,'contributions':[p.token for p in products]}
    previous=cache.read(directory)
    if previous:
        old={row['directory'] for row in previous.record['definition']['contributions']}
        new={str(p.directory) for p in products}
        for path in sorted(old-new): print(f'No longer selected for {module} {kind}: {path}',flush=True)
    product=cache.matching(directory,definition)
    if product is None:
        print(f'Combining {len(products)} cached {module} {method or "CM"} {kind} source contributions',flush=True)
        arrays=merge([p.arrays() for p in products],kind)
        product=_numeric_product(directory,definition,arrays,{'source_files':len(products),'events':int(arrays['events'])})
    return product


def diagnostic_products(config,module,info,combined):
    import numpy as np
    from utils.plot_statistics import joint_covariance
    from utils.plot_fits import fit_gaussian,fit_mip,json_safe
    paths=resolve_paths(config)['target'];out={}
    for method in config.methods:
        row={}
        if ('covariance',method) in combined:
            adc,cm=combined[('covariance',method)],combined[('cm',None)]
            definition={'adc':adc.token,'cm':cm.token,'formula':'centered pairwise population covariance, N denominator'}
            directory=Path(paths['covariances'])/module/method_directory(config,method)
            cov=cache.matching(directory,definition)
            if cov is None: cov=_numeric_product(directory,definition,joint_covariance(adc.arrays(),cm.arrays()),{})
            row['covariance']=cov
        if ('histograms',method) in combined:
            hist=combined[('histograms',method)];row['histograms']=hist
            if 'spectra' in config.plotting['plot_groups']:
                definition={'histograms':hist.token,'gaussian_initial':config.plotting['plot_gaussian_initial'],
                    'noise_window':config.plotting['plot_noise_window'],'mip_fit':config.plotting['plot_mip_fit'],
                    'mip_fit_range':config.plotting['plot_mip_fit_range'],'maxfev':config.plotting['plot_spectrum_maxfev'],
                    'fit_code_sha256':spectrum_code_hash(ROOT_DIR/'utils/plot_fits.py')}
                directory=hist.directory/'spectrum_fits';fit=cache.matching(directory,definition)
                if fit is None:
                    data=hist.arrays();settings=config.plotting
                    gaussian=fit_gaussian(data['adc_edges'],data['adc'],settings['plot_noise_window'],settings['plot_gaussian_initial'],settings['plot_spectrum_maxfev'])
                    curves,mip=fit_mip(data['mip_edges'],data['mip'],settings['plot_mip_fit_range'],settings['plot_spectrum_maxfev']) if settings['plot_mip_fit'] else ({},{'success':False,'message':'MIP fit explicitly disabled'})
                    summary=json_safe({'gaussian':gaussian,'mip':mip})
                    def build(work,curves=curves,summary=summary):
                        np.savez_compressed(work/'data.npz',**curves)
                        (work/'fit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
                        return summary
                    fit=cache.publish(directory,definition,build)
                row['spectrum_fits']=fit
        out[method]=row
    if set(config.plotting['plot_groups']) & {'noise_model','cell_area'}:
        attach_fits(config, module, info, out)
    return out


def run(config):
    from threadpoolctl import threadpool_limits
    from utils.plot_statistics import initialize
    from utils.plot_render import render_module
    print('Checking prepared/application provenance for plots',flush=True)
    modules=resolve_inputs(config)
    ROOT=initialize()
    if ROOT.IsImplicitMTEnabled(): ROOT.DisableImplicitMT()
    ROOT.EnableImplicitMT(config.resources['cpus'])
    for module,info in modules.items():
        files=contributions(config,module,info)
        combined={key:combine(config,module,key,value) for key,value in files.items()}
        with threadpool_limits(limits=config.resources['cpus'],user_api='blas'):
            products=diagnostic_products(config,module,info,combined)
            render_module(config,module,info,products)
    print(f'Plots ready: {resolve_paths(config)["target"]["plots"]}',flush=True)

