"""Array-only ports of the reference covariance and weighted-chi-square spectrum fits."""
from __future__ import annotations

from functools import lru_cache
import warnings


def json_safe(value):
    import numpy as np
    if isinstance(value, dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def noise_model(covariance, counts, rank, maxiter, *, gtol, ftol):
    """Reference objective/gradient with explicit, stricter L-BFGS-B tolerances."""
    import numpy as np
    from scipy.optimize import minimize
    if any(not np.isfinite(value) or value <= 0 for value in (gtol, ftol)):
        raise ValueError('Noise-fit gtol and ftol must be finite and positive')
    options={'maxiter':maxiter,'gtol':float(gtol),'ftol':float(ftol)}
    n=len(covariance)
    selected=np.flatnonzero(np.isfinite(np.diag(covariance)) & (np.diag(counts)>=2))
    arrays={name:np.full((n,n),np.nan) for name in ('model','coherent','incoherent','residual')}
    arrays['loadings']=np.full((n,rank),np.nan)
    if len(selected)<=rank:
        return arrays,{'success':False,'message':'Too few observed channels for requested rank','rank':rank,
                       'status':None,'optimizer_options':options}
    cov=covariance[np.ix_(selected,selected)];cov=.5*(cov+cov.T)
    valid=np.isfinite(cov) & (counts[np.ix_(selected,selected)]>=2)
    initial_cov=np.where(valid,cov,0.)
    values,vectors=np.linalg.eigh(initial_cov);order=np.argsort(values)[::-1][:rank]
    L0=vectors[:,order]*np.sqrt(np.maximum(values[order],0.))[None,:]
    d0=np.maximum(np.diag(initial_cov-L0@L0.T),0.)
    size=len(selected);normalization=int(valid.sum())
    def objective(params):
        d=params[:size];L=params[size:].reshape(size,rank)
        residual=np.where(valid,np.diag(d)+L@L.T-cov,0.)
        return float(np.sum(residual*residual)/normalization),np.r_[2*np.diag(residual)/normalization,(4*residual@L/normalization).ravel()]
    result=minimize(objective,np.r_[d0,L0.ravel()],jac=True,method='L-BFGS-B',
                    bounds=[(0,None)]*size+[(None,None)]*(size*rank),options=options)
    d=np.maximum(result.x[:size],0.);L=result.x[size:].reshape(size,rank)
    L=L[:,np.argsort(np.sum(L*L,axis=0))[::-1]]
    for j in range(rank):
        if L[np.argmax(np.abs(L[:,j])),j]<0: L[:,j]*=-1
    coherent=L@L.T;incoherent=np.diag(d);model=coherent+incoherent
    for name,block in [('model',model),('coherent',coherent),('incoherent',incoherent),('residual',cov-model)]:
        arrays[name][np.ix_(selected,selected)]=block
    arrays['loadings'][selected]=L
    total=float(np.trace(cov))
    summary={'success':bool(result.success),'status':int(result.status),'message':str(result.message),'rank':rank,
             'optimizer_options':options,
             'iterations':int(result.nit),'evaluations':int(result.nfev),
             'mse':objective(result.x)[0], 'observed_channels':selected.tolist(),
             'trace_total':total,'trace_coherent':float(np.trace(coherent)),
             'coherent_fraction':float(np.trace(coherent)/total) if total>0 else None}
    return arrays,json_safe(summary)


def gaussian(x, amplitude, mean, sigma):
    import numpy as np
    return amplitude*np.exp(-.5*((x-mean)/sigma)**2)


def fit_gaussian(edges,counts,window,initial,maxfev):
    import numpy as np
    from scipy.optimize import curve_fit
    x=.5*(edges[1:]+edges[:-1]);mask=(counts>0)&(x>=window[0])&(x<=window[1])
    if mask.sum()<=3: return {'success':False,'message':'Too few populated Gaussian-fit bins'}
    try:
        parameters,errors=curve_fit(gaussian,x[mask],counts[mask],p0=initial,
                                    sigma=np.sqrt(counts[mask]),absolute_sigma=True,maxfev=maxfev)
        parameters[2]=abs(parameters[2])
        if not np.all(np.isfinite(parameters)) or parameters[2]<=0:
            raise ValueError('Gaussian fit returned invalid parameters')
        residual=(counts[mask]-gaussian(x[mask],*parameters))/np.sqrt(counts[mask])
        return json_safe({'success':True,'parameters':parameters,'errors':np.sqrt(np.diag(errors)),
                          'chi2':float(residual@residual),'ndof':int(mask.sum()-3),
                          'objective':'weighted chi-square; positive bins in configured noise window'})
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        return {'success':False,'message':str(exc)}


@lru_cache(maxsize=1)
def _landau_lookup():
    import numpy as np
    from scipy.integrate import quad,IntegrationWarning
    grid=np.unique(np.r_[np.linspace(-6.,15.,700),np.linspace(15.,400.,500)])
    def pdf(x):
        def integrand(t):
            if t<=0: return 0.
            exponent=-t*np.log(t)-x*t
            return 0. if exponent>650 else np.exp(exponent)*np.sin(np.pi*t)
        return quad(integrand,0.,80.,limit=300)[0]/np.pi
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',IntegrationWarning)
        values=np.asarray([pdf(x) for x in grid])
    return grid,np.maximum(values,0.)


def langau(x,location,width,sigma):
    import numpy as np
    from scipy.signal import fftconvolve
    grid=np.linspace(float(np.min(x))-40.-5*sigma,float(np.max(x))+40.+5*sigma,4000)
    dx=grid[1]-grid[0];lookup,values=_landau_lookup()
    landau=np.interp((grid-location)/width,lookup,values,left=0.,right=0.)/width
    half=max(5*sigma,5*dx);kernel_x=np.arange(-half,half+dx,dx)
    kernel=np.exp(-.5*(kernel_x/sigma)**2);kernel/=kernel.sum()*dx
    return np.interp(x,grid,fftconvolve(landau,kernel,mode='same')*dx,left=0.,right=0.)


def mip_components(x,parameters):
    from scipy.stats import norm
    A,mu,sigma,B,location,width,C=parameters
    return (A*norm.pdf(x,mu,sigma),B*langau(x,location+mu,width,sigma),
            C*langau(x,2*location+mu,width,sigma))


def fit_mip(edges,counts,fit_range,maxfev):
    import numpy as np
    from scipy.optimize import curve_fit
    x=.5*(edges[1:]+edges[:-1]);valid=(x>=fit_range[0])&(x<=fit_range[1])
    x=x[valid];y=counts[valid]
    if len(x)<=7 or np.count_nonzero(y)<8:
        return {},{'success':False,'message':'Too few populated bins for pedestal+single/double-MIP fit'}
    initial=[float(np.max(y)),0.,1.,float(np.max(y))*.1,9.,1.,float(np.max(y))*.01]
    low=[0.,-10.,.05,0.,.5,.05,0.];high=[np.inf,10.,10.,np.inf,60.,10.,np.inf]
    def model(t,*parameters): return sum(mip_components(t,parameters))
    try:
        parameters,cov=curve_fit(model,x,y,p0=initial,sigma=np.sqrt(np.maximum(y,1.)),
                                 absolute_sigma=True,bounds=(low,high),maxfev=maxfev)
        prediction=model(x,*parameters);residual=(y-prediction)/np.sqrt(np.maximum(y,1.))
        chi2=float(residual@residual);ndof=len(x)-7
        summary={'success':True,'parameter_names':['A_ped','mu_p','sigma_p','A_mip','mu_L','c_L','A_2mip'],
                 'parameters':parameters,'errors':np.sqrt(np.diag(cov)),'chi2':chi2,'ndof':ndof,
                 'chi2_ndof':chi2/ndof if ndof>0 else None,
                 'objective':'reference weighted chi-square; includes empty bins with unit uncertainty'}
        dense=np.linspace(float(edges[0]),float(edges[-1]),2000)
        components=mip_components(dense,parameters)
        quality=[]
        if parameters[4]<=.5+.0595 or parameters[4]>=60.-.0595: quality.append('Landau location near parameter bound')
        if ndof<=0 or chi2/ndof>10: quality.append('Poor chi-square per degree of freedom')
        summary['fit_quality']={'valid':not quality,'warnings':quality}
        peak_grid=np.linspace(float(x.min()),float(x.max()),4000)
        summary['one_mip_peak']=float(peak_grid[int(np.argmax(mip_components(peak_grid,parameters)[1]))])
        return {'x':dense,'pedestal':components[0],'single_mip':components[1],'double_mip':components[2]},json_safe(summary)
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        return {},{'success':False,'message':str(exc)}
