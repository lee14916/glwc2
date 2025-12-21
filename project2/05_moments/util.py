import os,h5py,warnings,pickle,functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from math import floor, log10
from scipy.optimize import leastsq, curve_fit, fsolve
from scipy.linalg import solve_triangular,cholesky
from inspect import signature
from IPython.display import display,HTML

np.seterr(invalid=['ignore','warn'][0])
np.set_printoptions(legacy='1.25')

mpl.style.use('default')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.figsize'] = [6.4*1.2,4.8*1.2]
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['lines.marker'] = 's'
mpl.rcParams['lines.linestyle'] = ''
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['errorbar.capsize'] = 6
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.top']=mpl.rcParams['ytick.right']=True
mpl.rcParams['xtick.direction']=mpl.rcParams['ytick.direction']='in'
mpl.rcParams['legend.fontsize'] = 16
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

__all__ = ['np','os','plt','h5py','pickle','pd','display']

#!============== Initialization ==============#
if True:
    flag_fast=False # If True, certain functions will be speeded up using approximations.
    path_fig_internal=None; path_pkl_internal=None; path_fig=None; path_pkl=None
    def setpath(scriptname):
        global path_fig_internal, path_pkl_internal, path_fig, path_pkl
        path_fig_internal=f'fig/{scriptname}/internal_ignore/'
        path_pkl_internal=f'pkl/{scriptname}/internal_ignore/'
        path_fig=f'fig/{scriptname}/reg_ignore/'
        path_pkl=f'pkl/{scriptname}/reg_ignore/'
        
        for path in [path_fig_internal, path_pkl_internal, path_fig, path_pkl]:
            os.makedirs(path,exist_ok=True)

#!============== small functions ==============#
if True:
    deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)
    npRound=lambda dat,n:np.round(np.array(dat).astype(float),n)
    
    # c2pt2meff=lambda c2pt:np.log(np.roll(c2pt,1,axis=0)/c2pt) # use this one should also change the definition for func_meff_nst
    c2pt2meff=lambda c2pt:np.log(c2pt/np.roll(c2pt,-1,axis=0))
    nvec2n2= lambda nvec:nvec[0]**2+nvec[1]**2+nvec[2]**2
    
    def symmetrizeRatio(tf2ratio):
        for tf in tf2ratio.keys():
            tf2ratio[tf]=(tf2ratio[tf]+tf2ratio[tf][:,::-1])/2
        return tf2ratio

    def fsolve2(func,x0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res=fsolve(func, x0)[0]
        return res if res!=x0 else np.NaN
    
    cfg2old=lambda cfg: cfg[1:]+'_r'+{'a':'0','b':'1','c':'2','d':'3'}[cfg[0]]
    cfg2new=lambda cfg: {'0':'a','1':'b','2':'c','3':'d'}[cfg[-1]] + cfg[:4]
    
    def moms2dic(moms):
        dic={}
        for i,mom in enumerate(moms):
            dic[tuple(mom)]=i
        return dic
    def moms2list(moms):
        return [list(mom) for mom in moms]
    def mom2str(mom):
        return ','.join([str(ele) for ele in mom])
    def str2mom(momstr):
        return [int(ele) for ele in momstr.split(',')]
    
    def decodeList(l):
        return [ele.decode() for ele in l]
    def removeDuplicates(l):
        l=list(set(l)); l.sort()
        return l
    
    def any2filename(t):
        if type(t)==str:
            return t
        if type(t)==list:
            return ';'.join(t)
        1/0
    def save_pkl(file,res):
        with open(file,'wb') as f:
            pickle.dump(res,f)
    def load_pkl(file):
        with open(file,'rb') as f:
            res=pickle.load(f)
        return res
    def save_pkl_internal(label,res):
        if path_pkl_internal is None:
            print('path_pkl_internal is None, stop saving')
            return False
        save_pkl(f'{path_pkl_internal}{any2filename(label)}.pkl',res)
        return True
    def load_pkl_internal(file):
        if path_pkl_internal is None:
            print('path_pkl_internal is None, stop loading')
            return None
        if not os.path.isfile(f'{path_pkl_internal}{any2filename(file)}.pkl'):
            return None
        res=load_pkl(f'{path_pkl_internal}{any2filename(file)}.pkl')
        return res
    def save_pkl_reg(label,res):
        save_pkl(f'{path_pkl}{label}.pkl',res)
    def load_pkl_reg(label):
        return load_pkl(f'{path_pkl}{label}.pkl')
    def clear_pkl_internal(file):
        if os.path.isfile(f'{path_pkl_internal}{any2filename(file)}.pkl'):
            os.remove(f'{path_pkl_internal}{any2filename(file)}.pkl')
    
    def find_t_cloest(t,ens):
        return round(t/ens2a[ens])
    def get_tfs(tmin,tmax,ens,dt=1):
        return range(find_t_cloest(tmin,ens),find_t_cloest(tmax,ens),dt)
    def cut_tf2ratio(tf2ratio,tfmax,tfmin=0):
        return {tf:tf2ratio[tf] for tf in tf2ratio.keys() if tfmin<=tf<=tfmax}
    
    def removeError(dat):
        t=np.mean(dat,axis=0)
        return dat*0 + t[None,...]

#!============== constants ==============#
if True:
    hbarc = 1/197.3
    m_proton,m_neutron=938.2721,939.5654

#!============== error analysis ==============#
if True:
    def propagateError(func,mean,cov):
        '''
        Linear propagation of uncertainty
        y = func(x) = func(mean) + A(x-mean) + nonlinear terms; x~(mean,cov)
        '''
        y_mean=func(mean)
        AT=[]
        for j in range(len(mean)):
            unit=np.sqrt(cov[j,j])/1000
            ej=np.zeros(len(mean)); ej[j]=unit
            AT.append((np.array(func(mean+ej))-y_mean)/unit)
        AT=np.array(AT)
        A=AT.T
        y_cov=A@cov@AT
        return (np.array(y_mean),np.sqrt(np.diag(y_cov)),y_cov)
    def getCDR(cov):
        errs=np.sqrt(np.diag(cov))
        rho=cov/np.outer(errs, errs)
        
        evals=np.linalg.eigvalsh(rho)
        eval_max=np.max(evals); eval_min=np.min(evals)
        kappa=eval_max/eval_min
        CDR=10*np.log10(kappa)
        return CDR
    def jackknife(dat,d:int=0,nmin:int=6000):
        n=len(dat)
        if flag_fast:
            d=n//300
        elif d==0:
            d=n//nmin
        d=max(d,1)
        
        if d!=1:
            def tfunc(dat):
                shape=dat.shape
                nLeft=(shape[0]//d)*d
                shape_new=(shape[0]//d,d)+shape[1:]
                return dat[:nLeft].reshape(shape_new).mean(axis=1)
            dat_run=tfunc(dat)
        else:
            dat_run=dat
        n=len(dat_run)
        return np.array([np.mean(np.delete(dat_run,i,axis=0),axis=0) for i in range(n)])
    def jackme(dat_jk):
        n=len(dat_jk)
        dat_mean=np.mean(dat_jk,axis=0)
        dat_err=np.sqrt(np.var(dat_jk,axis=0,ddof=0)*(n-1))
        return (dat_mean,dat_err)
    def jackme_un2str(dat_jk):
        mean,err=jackme(dat_jk)
        if len(dat_jk.shape)==1:
            return un2str(mean,err)
        elif len(dat_jk.shape)==2:
            return [un2str(m,e) for m,e in zip(mean,err)]
        1/0
    def jackmec(dat_jk):
        n=len(dat_jk)
        dat_mean=np.mean(dat_jk,axis=0)
        dat_cov=np.atleast_2d(np.cov(np.array(dat_jk).T,ddof=0)*(n-1))
        dat_err=np.sqrt(np.diag(dat_cov))
        return (dat_mean,dat_err,dat_cov)
    def jackmap(func,dat_jk):
        t=[func(dat) for dat in dat_jk]
        if type(t[0]) is tuple:
            return tuple([np.array([t[i][ind] for i in range(len(t))]) for ind in range(len(t[0]))])
        return np.array(t)
    def jackknife_pseudo(mean,cov,n):
        mean=np.array(mean); cov=np.array(cov)
        if len(mean.shape)==len(cov.shape)==0:
            mean=mean[None]; cov=cov[None,None]**2
        if len(mean.shape)==len(cov.shape)==1:
            cov=np.diag(cov**2)
        dat_ens=np.random.multivariate_normal(mean,cov*n,n)
        dat_jk=jackknife(dat_ens)
        # do transformation [pars_jk -> A pars_jk + B] to force pseudo mean and err exactly the same
        mean1,_,cov1=jackmec(dat_jk)
        A=np.sqrt(np.diag(cov)/np.diag(cov1))
        B=mean-A*mean1
        dat_jk=A[None,:]*dat_jk+B[None,:]
        return dat_jk
    
    def jackknife2(in_dat,in_func=lambda dat:np.mean(np.real(dat),axis=0),minNcfg:int=600,d:int=0,outputFlatten=False,sl_key=None,sl_saveQ=False):
        '''
        - in_dat: any-dimensional list of ndarrays. Each ndarray in the list has 0-axis for cfgs
        - in_func: dat -> estimator
        - Estimator: number or 1d-list/array or 2d-list/array or 1d-list of 1d-arrays
        - d: jackknife delete parameter
        ### return: mean,err,cov
        - mean,err: estimator's format reformatted to 1d-list of 1d-arrays
        - cov: 2d-list of 2d-arrays
        '''  
        getNcfg=lambda dat: len(dat) if type(dat)==np.ndarray else getNcfg(dat[0])
        n=getNcfg(in_dat)
        if flag_fast:
            d=n//300
        elif d==0:
            d=n//minNcfg
        d=max(d,1)
        
        # average ${d} cfgs
        if d!=1:
            def tfunc(dat):
                if type(dat)==list:
                    return [tfunc(ele) for ele in dat]
                shape=dat.shape
                nLeft=(shape[0]//d)*d
                shape_new=(shape[0]//d,d)+shape[1:]
                return dat[:nLeft].reshape(shape_new).mean(axis=1)
            dat=tfunc(in_dat)
        else:
            dat=in_dat
        
        # reformat output of in_func
        t=in_func(dat)
        if type(t) in [list,np.ndarray] and type(t[0]) in [list,np.ndarray]:
            lenList=[len(ele) for ele in t]
            func=lambda dat:np.concatenate(in_func(dat))
        elif type(t) in [list,np.ndarray] and type(t[0]) not in [list,np.ndarray]:
            lenList=[len(t)]
            func=lambda dat:np.array(in_func(dat))
        elif type(t) not in [list,np.ndarray]:
            lenList=[1]
            func=lambda dat:np.array([in_func(dat)])
        else:
            1/0
            
        # delete i 
        delete= lambda dat,i: np.delete(dat,i,axis=0) if type(dat)==np.ndarray else [delete(ele,i) for ele in dat]
        
        # jackknife     
        n=getNcfg(dat)
        Tn1=np.array([func(delete(dat,i)) for i in range(n)])
        # print(np.mean(func()))
        # print(np.sqrt(np.var(Tn1)*n))
        TnBar=np.mean(Tn1,axis=0)
        (tMean,tCov)=(TnBar, np.atleast_2d(np.cov(Tn1.T,ddof=0)*(n-1)))
        # Tn=func(dat); (mean,cov)=(n*Tn-(n-1)*TnBar, np.atleast_2d(np.cov(Tn1.T,ddof)*(n-1))) # bias improvement (not suitable for fit)
        tErr=np.sqrt(np.diag(tCov))
        
        # reformat
        mean=[];err=[];t=0
        for i in lenList:
            mean.append(tMean[t:t+i])
            err.append(tErr[t:t+i])
            t+=i
        cov=[];t=0
        for i in lenList:
            covI=[];tI=0
            for j in lenList:
                covI.append(tCov[t:t+i,tI:tI+j]);tI+=j
            cov.append(covI);t+=i
            
        ret=(mean,err,cov)
        return ret

    def superjackknife(dats_jk):
        Nens=len(dats_jk)
        Ncfgss=[len(dat) for dat in dats_jk]
        dats_jkmean=[np.mean(dat_jk,axis=0) for dat_jk in dats_jk]
        t=[[dats_jk[i] if i==j else np.repeat(dats_jkmean[j][None,:],Ncfgss[i],axis=0) for j in range(Nens)] for i in range(Nens)]
        return np.block(t)

    def jackMA(fits,propagateChi2=True):
        ''' 
        fits=[fit]; fit=(fit_label,pars_jk,chi2_jk,Ndof)
        '''
        if propagateChi2:
            temp=[(pars_jk
                ,np.exp(-chi2_jk/2+Ndof) # weights_jk
                ) for fit_label,pars_jk,chi2_jk,Ndof in fits]
        else:
            temp=[(pars_jk
                ,np.exp(-np.mean(chi2_jk,axis=0)[:,None]/2+Ndof) # weights_jk
                ) for fit_label,pars_jk,chi2_jk,Ndof in fits]
        # print([weights_jk[0,0] for pars_jk,weights_jk in temp])
        weightsSum_jk=np.sum([weights_jk for _,weights_jk in temp],axis=0)
        pars_jk=np.sum([pars_jk*weights_jk/weightsSum_jk for pars_jk,weights_jk in temp],axis=0)
        probs_jk=np.transpose([weights_jk[:,0]/weightsSum_jk[:,0] for _,weights_jk in temp])
        return pars_jk,probs_jk

    def modelAvg(fits):
        '''
        fits=[fit]; fit=(pars_mean,pars_err,chi2,Ndof)
        '''
        weights=np.exp([-chi2/2+Ndof for pars_mean,pars_err,chi2,Ndof in fits])
        probs=weights/np.sum(weights)
        pars_mean_MA=np.sum(np.array([pars_mean for pars_mean,pars_err,chi2,Ndof in fits])*probs[:,None],axis=0)
        pars_err_MA=np.sqrt(np.sum(np.array([pars_err**2+pars_mean**2 for pars_mean,pars_err,chi2,Ndof in fits])*probs[:,None],axis=0)-pars_mean_MA**2)
        return (pars_mean_MA,pars_err_MA,probs)
    def jackMA2(fits): # doing model average after jackknife
        temp=[]
        for pars_jk,chi2_jk,Ndof in fits:
            pars_mean,pars_err=jackme(pars_jk)
            chi2_mean,chi2_err=jackme(chi2_jk)
            temp.append((pars_mean,pars_err,chi2_mean[0],Ndof))
        # print([np.exp(-chi2/2+Ndof) for pars_mean,pars_err,chi2,Ndof in temp])
        return modelAvg(temp)

    # uncertainty to string: taken from https://stackoverflow.com/questions/6671053/python-pretty-print-errorbars
    def un2str(x, xe, precision=2, forceResult = None):
        """pretty print nominal value and uncertainty

        x  - nominal value
        xe - uncertainty
        precision - number of significant digits in uncertainty

        returns shortest string representation of `x +- xe` either as
            x.xx(ee)e+xx
        or as
            xxx.xx(ee)"""
        # base 10 exponents
        x_exp = int(floor(log10(np.abs(x))))
        xe_exp = int(floor(log10(xe)))

        # uncertainty
        un_exp = xe_exp-precision+1
        un_int = round(xe*10**(-un_exp))

        # nominal value
        no_exp = un_exp
        no_int = round(x*10**(-no_exp))

        # format - nom(unc)exp
        fieldw = x_exp - no_exp
        
        if fieldw<0 and forceResult!=1:
            return un2str(x, xe, precision+1,forceResult=forceResult)
        if fieldw>=0:
            fmt = '%%.%df' % fieldw
            result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)
        else:
            result1 = None

        # format - nom(unc)
        fieldw = max(0, -no_exp)
        fmt = '%%.%df' % fieldw
        result2 = (fmt + '(%.0f)') % (no_int*10**no_exp, un_int*10**max(0, un_exp))
        if un_exp<0 and un_int*10**un_exp>=1:
            fmt2= '(%%.%df)' % (-un_exp)
            result2 = (fmt + fmt2) % (no_int*10**no_exp, un_int*10**un_exp)
        
        if forceResult is not None:
            return [result1,result2][forceResult]

        # return shortest representation
        if len(result2) <= len(result1):
            return result2
        else:
            return result1

#!============== fit (basic) ==============#
if True:
    def jackfit(fitfunc,y_jk,pars0,mask=None,parsExtra_jk=None,priors=[],**kargs):
        '''
        return=pars_jk,chi2_jk,Ndof,Nwarning
        priors=[(ind of par, mean, width)]
        '''
        with warnings.catch_warnings(record=True) as list_warnings:
            warnings.simplefilter("always")

            y_mean,_,y_cov=jackmec(y_jk)
            if mask is not None:
                if mask == 'uncorrelated':
                    y_cov=np.diag(np.diag(y_cov))
                else:
                    y_cov=y_cov*mask
                
            cho_L_Inv = np.linalg.inv(cholesky(y_cov, lower=True)) # y_cov^{-1}=cho_L_Inv^T@cho_L_Inv
            if parsExtra_jk is None:
                if len(priors)==0:
                    fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(pars)-y_mean)
                else:
                    fitfunc_wrapper=lambda pars: np.concatenate([cho_L_Inv@(fitfunc(pars)-y_mean),[(pars[ind]-mean)/width for ind,mean,width in priors]])
            else:
                parsExtra_mean=list(np.mean(parsExtra_jk,axis=0))
                if len(priors)==0:
                    fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(list(pars)+parsExtra_mean)-y_mean)
                else:
                    fitfunc_wrapper=lambda pars: np.concatenate([cho_L_Inv@(fitfunc(list(pars)+parsExtra_mean)-y_mean),[(pars[ind]-mean)/width for ind,mean,width in priors]])
            pars_mean,pars_cov=leastsq(fitfunc_wrapper,pars0,full_output=True,**kargs)[:2]
            
            if flag_fast == "FastFit": # Generate pseudo jackknife resamples from the single fit rather than doing lots of fits
                n=len(y_jk)
                pars_jk=jackknife_pseudo(pars_mean,pars_cov,n)
            else:
                if parsExtra_jk is None:
                    if len(priors)==0:
                        def func(dat):
                            fitfunc_wrapper2=lambda pars: cho_L_Inv@(fitfunc(pars)-dat)
                            pars=leastsq(fitfunc_wrapper2,pars_mean,**kargs)[0]
                            return pars
                    else:
                        def func(dat):
                            fitfunc_wrapper2=lambda pars: np.concatenate([cho_L_Inv@(fitfunc(pars)-dat),[(pars[ind]-mean)/width for ind,mean,width in priors]])
                            pars=leastsq(fitfunc_wrapper2,pars_mean,**kargs)[0]
                            return pars
                    pars_jk=jackmap(func,y_jk)
                else:
                    if len(priors)==0:
                        pars_jk=np.array([leastsq(lambda pars: cho_L_Inv@(fitfunc(list(pars)+list(parsExtra))-y),pars_mean,**kargs)[0] for y,parsExtra in zip(y_jk,parsExtra_jk)])
                    else:
                        pars_jk=np.array([leastsq(lambda pars: np.concatenate([cho_L_Inv@(fitfunc(list(pars)+list(parsExtra))-y),[(pars[ind]-mean)/width for ind,mean,width in priors]]),pars_mean,**kargs)[0] for y,parsExtra in zip(y_jk,parsExtra_jk)])
            chi2_jk=np.array([[np.sum(fitfunc_wrapper(pars)**2)] for pars in pars_jk])
            Ndata=len(y_mean); Npar=len(pars0); Ndof=Ndata-Npar
            
            Nwarning = len(list_warnings)
            for w in list_warnings:
                warnings.showwarning(message=w.message,category=w.category,filename=w.filename,lineno=w.lineno,file=w.file,line=w.line)
        return pars_jk,chi2_jk,Ndof,Nwarning
    
    def find_fitmax(dat,threshold=0.2):
        mean,err=jackme(dat)
        rela=np.abs(err/mean)
        temp=[(i,rela) for i,rela in enumerate(rela) if rela>0.2 and i!=0]
        fitmax=temp[0][0]-1 if len(temp)!=0 else len(mean)-1
        return fitmax
    
    def doFit_const(y_jk,corrQ=True):
        Ndata=y_jk.shape[1]
        if Ndata==1:
            return y_jk,np.zeros((len(y_jk),1)),0
        def fitfunc(pars):
            return list(pars)*Ndata
        pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,[np.mean(y_jk)],mask=None if corrQ else 'uncorrelated')
        return pars_jk,chi2_jk,Ndof
    
    def doFit_linear(xs,y_jk,corrQ=True):
        def fitfunc(pars):
            c0,c1=pars
            return c1*xs+c0
        pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,[np.mean(y_jk),0],mask=None if corrQ else 'uncorrelated')
        return pars_jk,chi2_jk,Ndof
    
    def fits2text(fits):
        text=[]
        pars_jk,probs_jk=jackMA(fits)
        Npar=pars_jk.shape[1]
        Ncut=Npar if Npar<20 else 1
        means,errs=jackme(pars_jk[:,:Ncut])
        probs_mean,probs_err=jackme(probs_jk)
        ind_mpf=np.argmax(probs_mean)
        fitlabel,pars_jk,chi2_jk,Ndof = fits[ind_mpf]
        chi2=np.mean(chi2_jk)
        text.append(f'Model average: pars={[un2str(mean,err) for mean,err in zip(means,errs)]}, most probable fitlabel: {fits[ind_mpf][0]}, prob={int(probs_mean[ind_mpf]*100)}%, chi2/Ndof={int(chi2*10)/10}/{Ndof}={int(chi2/Ndof*10)/10};')
        for i,fit in enumerate(fits):
            fitlabel,pars_jk,chi2_jk,Ndof = fit
            means,errs=jackme(pars_jk[:,:Ncut]); mean_chi2,err_chi2=jackme(chi2_jk)
            if Ndof!=0:
                text.append(f'fitlabel={fitlabel}, pars={[un2str(mean,err) for mean,err in zip(means,errs)]}, prob={int(probs_mean[i]*100)}%, chi2/Ndof={int(mean_chi2[0]*10)/10}/{Ndof}={int(mean_chi2[0]/Ndof*10)/10};')
            else:
                text.append(f'fitlabel={fitlabel}, pars={[un2str(mean,err) for mean,err in zip(means,errs)]}, prob={int(probs_mean[i]*100)}%, Ndof=0;')
        return text
    
    def decorator_fits(func):
        @functools.wraps(func)
        def wrapper(*args, label=None, overwrite=False, **kwargs):
            if label is not None and not overwrite:
                res=load_pkl_internal(label)
                if res is not None:
                    return res
            res = func(*args, **kwargs)
            if label is not None:
                if save_pkl_internal(label,res):
                    text=fits2text(res)
                    with open(f'{path_pkl_internal}{any2filename(label)}.txt','w') as f:
                        f.write('\n'.join(text))
            return res
        return wrapper

    @decorator_fits
    def doFit_continuumExtrapolation(ens2dat,lat_a2s_plt=None):
        enss=list(ens2dat.keys()); enss.sort()
        lat_a2s=[ens2a[ens]**2 for ens in enss]
        dat=[ens2dat[ens][:,None] for ens in enss]
        
        t=superjackknife(dat)
        fits=[]
        fitlabel='const'
        pars_jk,chi2_jk,Ndof=doFit_const(t)
        if lat_a2s_plt is not None:
            pars_jk=np.array([0*np.array(lat_a2s_plt)+pars[0] for pars in pars_jk])
        fits.append([fitlabel,pars_jk,chi2_jk,Ndof])
        fitlabel='linear'
        pars_jk,chi2_jk,Ndof=doFit_linear(np.array(lat_a2s),t)
        if lat_a2s_plt is not None:
            pars_jk=np.array([pars[1]*np.array(lat_a2s_plt)+pars[0] for pars in pars_jk])
        fits.append([fitlabel,pars_jk,chi2_jk,Ndof])
        return fits
#!============== fit (2pt) ==============#
if True:    
    @decorator_fits
    def doFit_2pt(dat,tmins,func,pars0,downSampling=1,corrQ=True,debugQ=False):
        tmax=find_fitmax(dat)
        fits=[]
        for tmin in tmins:
            ts=np.arange(tmin,tmax,downSampling)
            def fitfunc(pars):
                return func(ts,*pars)
            y_jk=dat[:,ts]
            pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
            if Nwarning or debugQ:
                print(f'[Nwarning={Nwarning}] Npar={len(pars0)}, tmin={tmin}')
            pars0=np.mean(pars_jk,axis=0)
            fits.append([tmin,pars_jk,chi2_jk,Ndof])
        return fits
    
    def decorator_fits_2pt(func):
        @functools.wraps(func)
        def wrapper(*args, label=None, overwrite=False, **kwargs):
            if label is not None and not overwrite:
                res=load_pkl_internal(label)
                if res is not None:
                    return res
            res = func(*args, **kwargs)
            if label is not None:
                if save_pkl_internal(label,res):
                    text=[]
                    for i,fits in enumerate(res):
                        text.append(f'{i+1} state fit')
                        text+=fits2text(fits)
                        text.append('\n')
                    with open(f'{path_pkl_internal}{any2filename(label)}.txt','w') as f:
                        f.write('\n'.join(text))
            return res
        return wrapper

    func_c2pt_1st=lambda t,E0,c0: c0*np.exp(-E0*t)
    func_c2pt_2st=lambda t,E0,c0,dE1,rc1: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t))
    func_c2pt_3st=lambda t,E0,c0,dE1,rc1,dE2,rc2: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t) + rc2*np.exp(-dE2*t))
    func_meff_1st=lambda t,E0: np.log(func_c2pt_1st(t,E0,1)/func_c2pt_1st(t+1,E0,1))
    func_meff_2st=lambda t,E0,dE1,rc1: np.log(func_c2pt_2st(t,E0,1,dE1,rc1)/func_c2pt_2st(t+1,E0,1,dE1,rc1))
    func_meff_3st=lambda t,E0,dE1,rc1,dE2,rc2: np.log(func_c2pt_3st(t,E0,1,dE1,rc1,dE2,rc2)/func_c2pt_3st(t+1,E0,1,dE1,rc1,dE2,rc2))
    @decorator_fits_2pt
    def doFit_meff_nst(meff,tminss,pars0,downSampling=1,corrQ=True,debugQ=False):
        Nst=len(tminss)
        fits_1st=doFit_2pt(meff,tminss[0],func_meff_1st,pars0[:1],downSampling=downSampling,corrQ=corrQ,debugQ=debugQ)
        if Nst==1:
            return [fits_1st]
        
        pars_jk,probs_jk=jackMA(fits_1st)
        pars0[:1]=np.mean(pars_jk,axis=0)
        fits_2st=doFit_2pt(meff,tminss[1],func_meff_2st,pars0[:3],downSampling=downSampling,corrQ=corrQ,debugQ=debugQ)
        if Nst==2:
            return [fits_1st,fits_2st]
        
        pars_jk,probs_jk=jackMA(fits_2st)
        pars0[:3]=np.mean(pars_jk,axis=0)
        fits_3st=doFit_2pt(meff,tminss[2],func_meff_3st,pars0[:5],downSampling=downSampling,corrQ=corrQ,debugQ=debugQ)
        if Nst==3:
            return [fits_1st,fits_2st,fits_3st]
#!============== fit (3pt) ==============#
if True:
    @decorator_fits
    def doFit_3pt_band(tf2ratio,tcmins,downSampling=1,corrQ=True):
        tfs=list(tf2ratio.keys()); tfs.sort()
        fits=[]
        for tf in tfs:
            for tcmin in tcmins:
                tcs_fit=np.arange(tcmin,tf-tcmin+1,downSampling)
                if len(tcs_fit)==0:
                    continue
                y_jk=tf2ratio[tf][:,tcs_fit]
                pars_jk,chi2_jk,Ndof=doFit_const(y_jk,corrQ=corrQ)
                fits.append([(tf,tcmin),pars_jk,chi2_jk,Ndof])
        return fits
    def doWA_band(fits,tf_min=None,tf_max=None,tcmin=None,corrQ=True):
        '''
        return: pars_jk,chi2_jk,Ndof,tf_min,tf_max,tcmin
        '''
        tfs=[fit[0][0] for fit in fits]
        tcmins=[fit[0][1] for fit in fits]
        if tf_min is None:
            tf_min=min(tfs)
        if tf_max is None:
            tf_max=max(tfs)
        if tcmin is None:
            tcmin=min(tcmins)
        fits=[fit for fit in fits if tf_min<=fit[0][0]<=tf_max and fit[0][1]==tcmin]
        y_jk=np.transpose([fit[1][:,0] for fit in fits])
        pars_jk,chi2_jk,Ndof=doFit_const(y_jk)
        return pars_jk,chi2_jk,Ndof,tf_min,tf_max,tcmin
    
    def doMA_3pt(fits,tfmin_min=None,tfmin_max=None,tcmin_min=None,tcmin_max=None,threshold=0):
        '''
        return: pars_jk,chi2_jk,Ndof,tf_min,tf_max,tcmin
        '''
        tfmins=[fit[0][0] for fit in fits]
        tcmins=[fit[0][1] for fit in fits]
        if tfmin_min is None:
            tfmin_min=min(tfmins)
        if tfmin_max is None:
            tfmin_max=max(tfmins)
        if tcmin_min is None:
            tcmin_min=min(tcmins)
        if tcmin_max is None:
            tcmin_max=max(tcmins)
            
        fits=[fit for fit in fits if tfmin_min<=fit[0][0]<=tfmin_max and tcmin_min<=fit[0][1]<=tcmin_max]
        pars_jk,probs_jk=jackMA(fits)
        if threshold!=0:
            probs=np.mean(probs_jk,axis=0)
            inds=[i for i in range(len(probs)) if probs[i]>=threshold]
            fits=fits[inds]
            pars_jk,probs_jk=jackMA(fits)
        fitlabels=[fit[0] for fit in fits]
        return pars_jk,probs_jk,fitlabels
    
    @decorator_fits
    def doFit_3pt_1st(tf2ratio_para,tfmins,tcmins,pars0=None,downSampling=[1,1],symmetrizeQ=False,corrQ=True):
        tf2ratio=tf2ratio_para.copy()
        tfs=list(tf2ratio.keys()); tfs.sort()
        if symmetrizeQ:
            symmetrizeRatio(tf2ratio)
        if pars0 is None:
            tfmin=np.min(tfs)
            g=np.mean(tf2ratio[tfmin][:,tfmin//2])
            pars0=[g]
        fits=[]
        for tfmin in tfmins:
            for tcmin in tcmins:
                if tfmin<tcmin*2:
                    continue
                tfs_fit=[tf for tf in tfs if tfmin<=tf and tf%downSampling[0]==tfmin%downSampling[0]]
                tf2tcs_fit={tf:np.arange(tcmin,tf//2+1,downSampling[1]) if symmetrizeQ else np.arange(tcmin,tf-tcmin+1,downSampling[1])  for tf in tfs_fit}
                y_jk=np.concatenate([tf2ratio[tf][:,tf2tcs_fit[tf]] for tf in tfs_fit],axis=1)
                Ndata=y_jk.shape[1]
                def fitfunc(pars):
                    return list(pars)*Ndata
                pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
                fits.append([(tfmin,tcmin),pars_jk,chi2_jk,Ndof])
        return fits
    
    @decorator_fits
    def doFit_3pt_sum(tf2ratio,tfmins,tcmins,pars0=None,downSampling=1,corrQ=True):
        tfs=list(tf2ratio.keys()); tfs.sort()
        if pars0 is None:
            tfmin=np.min(tfs)
            g=np.mean(tf2ratio[tfmin][:,tfmin//2])
            pars0=[g,0]
        fits=[]
        for tfmin in tfmins:
            for tcmin in tcmins:
                if tfmin<tcmin*2:
                    continue
                tfs_fit=[tf for tf in tfs if tfmin<=tf and tf%downSampling==tfmin%downSampling]
                y_jk=np.transpose([np.sum(tf2ratio[tf][:,tcmin:tf+1-tcmin],axis=1) for tf in tfs_fit])
                def fitfunc(pars):
                    g,c=pars
                    t=np.array([g*tf+c for tf in tfs_fit])
                    return t
                pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
                fits.append([(tfmin,tcmin),pars_jk,chi2_jk,Ndof])
        return fits
        
    func_c3pt_2st=lambda tf,tc,E0a,E0b,a00,dE1a,dE1b,ra01,ra10,ra11: a00*np.exp(-E0a*(tf-tc))*np.exp(-E0b*tc)*(1 + ra01*np.exp(-dE1b*tc) + ra10*np.exp(-dE1a*(tf-tc)) + ra11*np.exp(-dE1a*(tf-tc))*np.exp(-dE1b*tc)) \
        if a00!=0 else np.exp(-E0a*(tf-tc))*np.exp(-E0b*tc)*(ra01*np.exp(-dE1b*tc) + ra10*np.exp(-dE1a*(tf-tc)) + ra11*np.exp(-dE1a*(tf-tc))*np.exp(-dE1b*tc))
    func_ratio_2st=lambda tf,tc,g,dE1,rc1,ra01,ra11:func_c3pt_2st(tf,tc,0,0,g,dE1,dE1,ra01,ra01,ra11)/func_c2pt_2st(tf,0,1,dE1,rc1)
    @decorator_fits
    def doFit_3ptSym_2st2step(tf2ratio_para,tfmins,tcmins,pars_jk_meff2st,pars0=None,downSampling=[1,1],symmetrizeQ=False,corrQ=True,debugQ=False):
        tf2ratio=tf2ratio_para.copy()
        tfs=list(tf2ratio.keys()); tfs.sort()
        if symmetrizeQ:
            for tf in tfs:
                tf2ratio[tf]=(tf2ratio[tf]+tf2ratio[tf][:,::-1])/2
        if pars0 is None:
            tfmin=np.min(tfs)
            g=np.mean(tf2ratio[tfmin][:,tfmin//2])
            pars0=[g,0,0]
        
        fits=[]
        for tfmin in tfmins:
            for tcmin in tcmins:
                if tfmin<tcmin*2:
                    continue
                
                tfs_fit=[tf for tf in tfs if tfmin<=tf and tf%downSampling[0]==tfmin%downSampling[0]]
                tf2tcs_fit={tf:np.arange(tcmin,tf//2+1,downSampling[1]) if symmetrizeQ else np.arange(tcmin,tf-tcmin+1,downSampling[1])  for tf in tfs_fit}
                
                y_jk=np.concatenate([tf2ratio[tf][:,tf2tcs_fit[tf]] for tf in tfs_fit],axis=1)
                def fitfunc(pars):
                    g,ra01,ra11, E0,dE1,rc1=pars
                    t=np.concatenate([func_ratio_2st(tf,tf2tcs_fit[tf],g,dE1,rc1,ra01,ra11) for tf in tfs_fit])
                    return t
                pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,pars0,parsExtra_jk=pars_jk_meff2st,mask=None if corrQ else 'uncorrelated')
                if Nwarning or debugQ:
                    print(f'[Nwarning={Nwarning}] tfmin={tfmin}, tcmin={tcmin}')
                fits.append([(tfmin,tcmin),pars_jk,chi2_jk,Ndof])
        return fits

#!============== table ==============#
if True:
    def dfs2html(dfs,titles=None):
        if type(dfs)==pd.core.frame.DataFrame:
            dfs=[dfs]; titles=[titles]
        if titles is None:
            titles=[None]*len(dfs)
        
        blocks = []
        for df,title in zip(dfs,titles):
            blocks.append(f"""
                <div>
                    <h3 style="text-align:center;">{title}</h3>
                    {df.to_html()}
                </div>
            """)

        html = f"""
        <div style="display:flex; gap:20px;">
            {''.join(blocks)}
        </div>
        """

        return HTML(html)

#!============== plot (basic) ==============#
if True:
    colors8=['r','g','b','orange','purple','brown','magenta','olive']
    fmts8=['s','o','d','^','v','<','>','*']
    
    colors16=['blue','orange','green','red','purple','brown','darkblue','olive','darkgreen','darkred','grey','tan','peru','magenta','gold','skyblue']
    fmts16=['o','v','^','<','>','d','s','h','*','H','p','8','X','P','D','.']
    
    def getFigAxs(Nrow,Ncol,Lrow=None,Lcol=None,scale=1,**kwargs):
        if (Lrow,Lcol)==(None,None):
            Lcol,Lrow=mpl.rcParams['figure.figsize']
            Lrow*=scale; Lcol*=scale
            # if (Nrow,Ncol)==(1,1):
            #     Lcol*=1.5; Lrow*=1.5
        fig, axs = plt.subplots(Nrow, Ncol, figsize=(Lcol*Ncol, Lrow*Nrow), squeeze=False,**kwargs)
        fig.align_ylabels()
        return fig, axs

    def addRowHeader(axs,rows,fontsize='xx-large',**kargs):
        pad=5
        for ax, row in zip(axs[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points', ha='right', va='center', fontsize=fontsize, **kargs)
            
    def addColHeader(axs,cols,fontsize='xx-large',**kargs):
        pad=5
        for ax, col in zip(axs[0,:], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline', fontsize=fontsize, **kargs)
            
    def addRefLine(ax,val,hv='h',color='grey',ls='--',marker='',label=None):
        axline={'h':ax.axhline,'v':ax.axvline}[hv]
        axline(val,color=color,ls=ls,marker=marker,label=label)

    def finalizePlot(file=None,closeQ=None):
        if closeQ is None:
            closeQ=False if file is None else True
        plt.tight_layout()
        if file!=None:
            if path_fig_internal is None:
                print('path_fig_internal is None')
                return
            plt.savefig(f'{path_fig_internal}{any2filename(file)}.pdf',bbox_inches="tight")
        if closeQ:
            plt.close()
    
    def makePDF(file,figs):
        assert(path_fig_internal is not None)
        pdf = PdfPages(f'{path_fig_internal}{any2filename(file)}.pdf')
        for fig in figs:
            pdf.savefig(fig,bbox_inches="tight")
        pdf.close()
            
    def makePlot_simpleComparison(xs,y_jk,xticklabels=None):
        fig, axs = getFigAxs(1,1)
        ax=axs[0,0]
        mean,err=jackme(y_jk)
        ax.errorbar(xs,mean,err,color='r',fmt='s')
        ax.set_xticks(xs)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        return fig,axs
    
    # def addPlot_continuumExtrapolation(ax,ens2dat,lat_a2s_plt=None,color='g',shows=['MA'],colors=None,prelabels=None):
    #     enss=list(ens2dat.keys()); enss.sort()
    #     lat_a2s=[ens2a[ens]**2 for ens in enss]
    #     lat_a2s_plt=np.arange(0,np.max(lat_a2s)*1.2,np.max(lat_a2s)/0.1) if lat_a2s_plt is None else lat_a2s_plt
        
    #     dat=[ens2dat[ens] for ens in enss]
    #     fits=doFit_continuumExtrapolation(dat,lat_a2s_plt=lat_a2s_plt)
        
    #     if colors is None and len(shows)==1:
    #         colors=[color]
    #     if colors is None and len(shows)>1:
    #         dic={'const':'r','linear':'b','MA':'orange'}
    #         colors=[dic[show] for show in shows]
    #     fitlabel2color={fitlabel:colors[i] for i,fitlabel in enumerate(shows)}
        
    #     if prelabels is None:
    #         prelabels=[None]*len(shows)
    #     fitlabel2prelabel={fitlabel:prelabels[i] for i,fitlabel in enumerate(shows)} 
        
    #     mes=[jackme(ele) for ele in dat]
    #     plt_x=np.array(lat_a2s); plt_y=[me[0][0] for me in mes]; plt_yerr=[me[1][0] for me in mes]
    #     ax.errorbar(plt_x,plt_y,plt_yerr,color=color)
        
    #     for fit in fits:
    #         fitlabel,pars_jk,chi2_jk,Ndof=fit
    #         if fitlabel not in shows:
    #             continue
    #         mean,err=jackme(pars_jk)
    #         x=np.array(lat_a2s_plt); ymin=mean-err; ymax=mean+err
    #         ax.plot(x,mean,color=fitlabel2color[fitlabel],linestyle='--',marker='')
    #         prelabel=fitlabel2prelabel[fitlabel]
    #         label=None if prelabel is None else f'{prelabel}{un2str(mean[0],err[0])}' if prelabel.endswith('=') else prelabel
    #         ax.fill_between(x, ymin, ymax, color=fitlabel2color[fitlabel], alpha=0.1,label=label)
    #     for fitlabel in ['MA']:
    #         if fitlabel not in shows:
    #             continue
    #         pars_jk,probs_jk=jackMA(fits)
    #         mean,err=jackme(pars_jk)
    #         x=np.array(lat_a2s_plt); ymin=mean-err; ymax=mean+err
    #         ax.plot(x,mean,color=fitlabel2color[fitlabel],linestyle='--',marker='')
    #         prelabel=fitlabel2prelabel[fitlabel]
    #         label=None if prelabel is None else f'{prelabel}{un2str(mean[0],err[0])}' if prelabel.endswith('=') else prelabel
    #         ax.fill_between(x, ymin, ymax, color=fitlabel2color[fitlabel], alpha=0.1,label=label)
    
    def makePlot_continuumExtrapolation(matrix_dat,lat_a2s,lat_a2s_plt,matrix_fits,shows=['MA'],multiRowColQ=(False,False)):
        matrix_dat,matrix_fits={
            (False,False):([[matrix_dat]],[[matrix_fits]]),
            (False,True):([matrix_dat],[matrix_fits]),
            (True,False):([[dat] for dat in matrix_dat],[[fits] for fits in matrix_fits]),
            (True,True):(matrix_dat,matrix_fits),
        }[multiRowColQ]
        
        Nrow,Ncol=len(matrix_dat),len(matrix_dat[0])
        fig,axs=getFigAxs(Nrow,Ncol,sharex='col',sharey='row')
        for icol in range(Ncol):
            ax=axs[-1,icol]
            ax.set_xlabel(r'$a^2$ [fm$^2$]')
            
        for irow in range(Nrow):
            for icol in range(Ncol):
                ax=axs[irow,icol]
                dat=matrix_dat[irow][icol]; fits=matrix_fits[irow][icol]

                mes=[jackme(ele) for ele in dat]
                plt_x=np.array(lat_a2s); plt_y=[me[0][0] for me in mes]; plt_yerr=[me[1][0] for me in mes]
                ax.errorbar(plt_x,plt_y,plt_yerr,color='g')
                
                colors={'const':'r','linear':'b','MA':'orange'}
                
                for fit in fits:
                    fitlabel,pars_jk,chi2_jk,Ndof=fit
                    if fitlabel not in shows:
                        continue
                    mean,err=jackme(pars_jk)
                    x=np.array(lat_a2s_plt); ymin=mean-err; ymax=mean+err
                    ax.plot(x,mean,color=colors[fitlabel],linestyle='--',marker='')
                    ax.fill_between(x, ymin, ymax, color=colors[fitlabel], alpha=0.1,label=f'{fitlabel}={un2str(mean[0],err[0])}')
                for fitlabel in ['MA']:
                    if fitlabel not in shows:
                        continue
                    pars_jk,probs_jk=jackMA(fits)
                    mean,err=jackme(pars_jk)
                    x=np.array(lat_a2s_plt); ymin=mean-err; ymax=mean+err
                    ax.plot(x,mean,color=colors[fitlabel],linestyle='--',marker='')
                    ax.fill_between(x, ymin, ymax, color=colors[fitlabel], alpha=0.1,label=f'{fitlabel}={un2str(mean[0],err[0])}')
                
                ax.legend(fontsize=16)
        return fig,axs
#!============== plot (2pt) ==============#
if True:
    def makePlot_2pt_SimoneStyle(meff,fitss,xunit=1,yunit=1,mN_exp=None,selection={},ylims='auto'):
        for _ in [0]:
            result={}
            fig, axd = plt.subplot_mosaic([['f1','f1','f1'],['f2','f2','f3']],figsize=(24,10))
            (ax1,ax2,ax3)=(axd[key] for key in ['f1','f2','f3'])
            ax1.set_xlabel(r'$t$ [fm]')
            ax2.set_xlabel(r'$t_{\mathrm{min}}$ [fm]')
            ax3.set_xlabel(r'$t_{\mathrm{min}}$ [fm]')
            ax1.set_ylabel(r'$m_N^{\mathrm{eff}}$ [GeV]')
            ax2.set_ylabel(r'$m_N$ [GeV]')
            ax3.set_ylabel(r'$E_1$ [GeV]')
            if ylims=='std_N':
                ylims=[[0.86,1.11],[0.86,1.11],[0,3.9]]
            if ylims!='auto':
                ax1.set_ylim(ylims[0]); ax2.set_ylim(ylims[1]); ax3.set_ylim(ylims[2])

            mean,err=jackme(meff)
            fitmax=find_fitmax(meff)
            
            tmin=1; tmax=fitmax+1
            plt_x=np.arange(tmin,tmax)*xunit; plt_y=mean[tmin:tmax]*yunit; plt_yerr=err[tmin:tmax]*yunit
            ax1.errorbar(plt_x,plt_y,plt_yerr,color='black',fmt='s')
            
            if mN_exp is not None:
                ax1.axhline(y=mN_exp,color='black',linestyle = '--', marker='')
                ax2.axhline(y=mN_exp,color='black',linestyle = '--', marker='', label=r'$m_N^{\mathrm{exp}}=$'+'%0.3f'%mN_exp)
            
            Nst=len(fitss)
            probThreshold=0.1
            chi2Size=12
            DNpar=1 # DNpar=1 if meffQ else 0
            percentage_shiftMultiplier=1.5
            
            if Nst==0:
                continue
            
            fitcase='1st'
            color='r'
            fits=fitss[0]    
            fitmins=[fit[0] for fit in fits]
            pars_jk,probs_jk=jackMA(fits)
            probs_mean=np.mean(probs_jk,axis=0)
            ind_mpf=np.argmax(np.mean(probs_jk,axis=0))  
            if fitcase in selection:
                ind_select=fitmins.index(selection[fitcase])
                result[fitcase]=fits[ind_select][1]
            else:
                result[fitcase]=pars_jk
            pars_mean,pars_err=jackme(result[fitcase])
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2,label=r'$m_N^{\mathrm{1st}}=$'+un2str(plt_y,plt_yerr))
            if ylims=='auto':
                ax1.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*40])
                ax2.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*30])
            for i,fit in enumerate(fits):
                fitmin,pars_jk,chi2_jk,Ndof=fit; prob=probs_mean[i]
                (pars_mean,pars_err)=jackme(pars_jk)
                chi2R=np.mean(chi2_jk)/Ndof
                if fitcase in selection:
                    showQ = i==ind_select
                else:
                    showQ = i==ind_mpf if probThreshold is None else prob>probThreshold
                plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
                ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='s',color=color,mfc='white' if showQ else None)
                ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold:
                    ax2.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
            
            if Nst==1:
                continue
            
            fitcase='2st'
            color='g'
            fits=fitss[1]    
            fitmins=[fit[0] for fit in fits]
            pars_jk,probs_jk=jackMA(fits)
            probs_mean=np.mean(probs_jk,axis=0)
            ind_mpf=np.argmax(np.mean(probs_jk,axis=0)) 
            if fitcase in selection:
                ind_select=fitmins.index(selection[fitcase])
                result[fitcase]=fits[ind_select][1]
            else:
                result[fitcase]=pars_jk
            t=np.transpose([result[fitcase][:,0],result[fitcase][:,0]+result[fitcase][:,2-DNpar]])
            pars_mean,pars_err=jackme(t)
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$m_N^{\mathrm{2st}}=$'+un2str(plt_y,plt_yerr))
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
            ax3.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_1^{\mathrm{2st}}=$'+un2str(plt_y,plt_yerr))
            if ylims=='auto':
                ax3.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*30])
            for i,fit in enumerate(fits):
                fitmin,pars_jk,chi2_jk,Ndof=fit; prob=probs_mean[i]
                t=pars_jk.copy()
                t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
                (pars_mean,pars_err)=jackme(t)
                chi2R=np.mean(chi2_jk)/Ndof
                if fitcase in selection:
                    showQ = i==ind_select
                else:
                    showQ = i==ind_mpf if probThreshold is None else prob>probThreshold
                plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
                ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='o',color=color,mfc='white' if showQ else None)
                ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold:
                    ax2.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
                
                plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
                ax3.errorbar(plt_x,plt_y,plt_yerr,fmt='o',color=color,mfc='white' if showQ else None)
                ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold:
                    ax3.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
                    
            if Nst==2:
                continue
            
            fitcase='3st'
            color='b'
            fits=fitss[2]    
            fitmins=[fit[0] for fit in fits]
            pars_jk,probs_jk=jackMA(fits)
            probs_mean=np.mean(probs_jk,axis=0)
            ind_mpf=np.argmax(np.mean(probs_jk,axis=0))    
            if fitcase in selection:
                ind_select=fitmins.index(selection[fitcase])
                result[fitcase]=fits[ind_select][1]
            else:
                result[fitcase]=pars_jk
            t=np.transpose([result[fitcase][:,0],result[fitcase][:,0]+result[fitcase][:,2-DNpar]])
            pars_mean,pars_err=jackme(t)
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$m_N^{\mathrm{3st}}=$'+un2str(plt_y,plt_yerr))
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
            ax3.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_1^{\mathrm{3st}}=$'+un2str(plt_y,plt_yerr))    
            for i,fit in enumerate(fits):
                fitmin,pars_jk,chi2_jk,Ndof=fit; prob=probs_mean[i]
                t=pars_jk.copy()
                t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
                (pars_mean,pars_err)=jackme(t)
                chi2R=np.mean(chi2_jk)/Ndof
                if fitcase in selection:
                    showQ = i==ind_select
                else:
                    showQ = i==ind_mpf if probThreshold is None else prob>probThreshold
                plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
                ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='d',color=color,mfc='white' if showQ else None)
                ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold:
                    ax2.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
                
                plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
                ax3.errorbar(plt_x,plt_y,plt_yerr,fmt='d',color=color,mfc='white' if showQ else None)
                ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center') 
                if probThreshold is not None and prob>probThreshold:
                    ax3.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
        
        ax2.legend(fontsize=16)
        ax3.legend(fontsize=16)
        return fig,axd,result           
#!============== plot (3pt) ==============#
if True:
    def makePlot_3pt(list_dic,shows=['rainbow','fit_band','fit_const','fit_sum','fit_2st'],colHeaders='auto',colors_rainbow=colors16,colors_fit=colors8):
        width_ratios=[3 if show in ['rainbow'] else 2 for show in shows]
        Nrow=len(list_dic); Ncol=len(shows)
        fig, axs = getFigAxs(Nrow,Ncol,Lrow=4,Lcol=6,sharex='col',sharey='row', gridspec_kw={'width_ratios': width_ratios})
        if colHeaders is not None:
            show2Header={'rainbow':r'ratio','midpoint':r'mid point','fit_band':r'const fit to each $t_s$',
                    'fit_const':r'const fit', 'fit_2st':r'2st fit', 'fit_sum':r'summation method',
                    'fit_const_prob':r'Fit Prob.','fit_sum_prob':r'Fit Prob.','fit_2st_prob':r'Fit Prob.'}
            addColHeader(axs,[show2Header[show] for show in shows] if colHeaders=='auto' else colHeaders)
            
        show2xlabel={'rainbow':r'$t_{\rm ins}-t_{s}/2$ [fm]','midpoint':r'$t_{s}^{\rm}$ [fm]','fit_band':r'$t_{s}^{\rm}$ [fm]',
                    'fit_const':r'$t_{s}^{\rm low}$ [fm]', 'fit_2st':r'$t_{s}^{\rm low}$ [fm]', 'fit_sum':r'$t_{s}^{\rm low}$ [fm]',
                    'fit_const_prob':r'Fit Prob.','fit_sum_prob':r'Fit Prob.','fit_2st_prob':r'Fit Prob.'}
        for i in range(Ncol):
            axs[-1,i].set_xlabel(show2xlabel[shows[i]])
        
        for irow in range(Nrow):
            dic=list_dic[irow]
            def setParameter(default,key):
                if type(default) is not list:
                    return dic[key] if key in dic else default                
                if key in dic:
                    res=dic[key]
                    for i in range(len(res)):
                        if res[i] is None:
                            res[i]=default[i]
                    return res
                return default
            def setParameter2(default,now):
                return default if now is None else now
            
            # general
            xunit=setParameter(1,'xunit')
            yunit=setParameter(1,'yunit')
            (tf2ratio,fits_band,fits_const,fits_sum,fits_2st)=setParameter([None,None,None,None,None],'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]')
            tfs_all=list(tf2ratio.keys()); tfs_all.sort()
            # rainbow
            (tfmin,tfmax,tcmin,dt)=setParameter([0,np.inf,1,1],'rainbow:[tfmin,tfmax,tcmin,dt]')
            tfs_rainbow=[tf for tf in tfs_all if tfmin<=tf<=tfmax and tf%dt==0 and tf>=tcmin*2]
            tcmin_rainbow=tcmin
            tfs_mid=setParameter(tfs_rainbow,'tfs_mid')
            # fit_band
            if fits_band is not None:
                tfs=removeDuplicates([fit[0][0] for fit in fits_band])
                tcmins=removeDuplicates([fit[0][1] for fit in fits_band])
                (tfmin,tfmax,tcmin_min,tcmin_max)=setParameter([min(tfs),max(tfs),min(tcmins),max(tcmins)],'fit_band:[tfmin,tfmax,tcmin_min,tcmin_max]')
                [dt]=setParameter([1],'fit_band:[dt]')
                tfs_band=[tf for tf in tfs if tfmin<=tf<=tfmax and tf%dt==0]
                tcmins_band=[tcmin for tcmin in tcmins if tcmin_min<=tcmin<=tcmin_max]
                fit_band_WA=setParameter(None,'fit_band_WA')
            else:
                tfs_band=[]
            # fit_const
            if fits_const is not None:
                fits=fits_const
                tfmins=removeDuplicates([fit[0][0] for fit in fits])
                tcmins=removeDuplicates([fit[0][1] for fit in fits])
                (tfmin_min,tfmin_max,tcmin_min,tcmin_max)=setParameter([min(tfmins),max(tfmins),min(tcmins),max(tcmins)],'fit_const:[tfmin_min,tfmin_max,tcmin_min,tcmin_max]')
                tfmins_const=[tfmin for tfmin in tfmins if tfmin_min<=tfmin<=tfmin_max]
                tcmins_const=[tcmin for tcmin in tcmins if tcmin_min<=tcmin<=tcmin_max]
                fit_const_MA=setParameter(None,'fit_const_MA')
            # fit_sum
            if fits_sum is not None:
                fits=fits_sum
                tfmins=removeDuplicates([fit[0][0] for fit in fits])
                tcmins=removeDuplicates([fit[0][1] for fit in fits])
                (tfmin_min,tfmin_max,tcmin_min,tcmin_max)=setParameter([min(tfmins),max(tfmins),min(tcmins),max(tcmins)],'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max]')
                tfmins_sum=[tfmin for tfmin in tfmins if tfmin_min<=tfmin<=tfmin_max]
                tcmins_sum=[tcmin for tcmin in tcmins if tcmin_min<=tcmin<=tcmin_max]
                fit_sum_MA=setParameter(None,'fit_sum_MA')
            # fit_2st
            if fits_2st is not None:
                fits=fits_2st
                tfmins=removeDuplicates([fit[0][0] for fit in fits])
                tcmins=removeDuplicates([fit[0][1] for fit in fits])
                (tfmin_min,tfmin_max,tcmin_min,tcmin_max)=setParameter([min(tfmins),max(tfmins),min(tcmins),max(tcmins)],'fit_2st:[tfmin_min,tfmin_max,tcmin_min,tcmin_max]')
                tfmins_2st=[tfmin for tfmin in tfmins if tfmin_min<=tfmin<=tfmin_max]
                tcmins_2st=[tcmin for tcmin in tcmins if tcmin_min<=tcmin<=tcmin_max]
                fit_2st_MA=setParameter(None,'fit_2st_MA')
            
            
            tfs_color=removeDuplicates(tfs_rainbow+tfs_band+tfs_mid)
            
            show='rainbow'
            if show in shows:
                ax=axs[irow,shows.index(show)]                
                for itf,tf in enumerate(tfs_rainbow):
                    mean,err=jackme(tf2ratio[tf])
                    tcs=np.arange(tcmin_rainbow,tf-tcmin_rainbow+1)
                    plt_x=(tcs-tf/2+0.05*(itf-len(tfs_rainbow)/2))*xunit; plt_y=mean[tcs]*yunit; plt_yerr=err[tcs]*yunit
                    itf_color=tfs_color.index(tf)
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_rainbow[itf_color%16],fmt=fmts16[itf_color%16])
                    
            show='fit_band'
            if show in shows and fits_band is not None:
                ax=axs[irow,shows.index(show)]   
                fits=fits_band
                if fit_band_WA is not None:
                    pars_jk,chi2_jk,Ndof,tf_min_WA,tf_max_WA,tcmin_WA=fit_band_WA
                    mean,err=jackme(pars_jk[:,0])
                    plt_x=[tf_min_WA*xunit,tf_max_WA*xunit]; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))  
                    ax.legend()
                    
                tcmins=removeDuplicates([fit[0][1] for fit in fits])
                for fit in fits:
                    (tf,tcmin),pars_jk,chi2_jk,Ndof=fit
                    if tf not in tfs_band or tcmin not in tcmins_band:
                        continue
                    itf=tfs_color.index(tf)
                    itcmin=tcmins.index(tcmin)
                    mfc=None
                    if fit_band_WA is not None and tf_min_WA<=tf<=tf_max_WA and tcmin==tcmin_WA:
                        mfc='white'
                    mean,err=jackme(pars_jk[:,0])
                    plt_x=(tf+itcmin*0.1)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_rainbow[itf%16],fmt=fmts16[itf%16],mfc=mfc)
            
            show='fit_const'
            fits=fits_const
            if show in shows and fits is not None:
                ax=axs[irow,shows.index(show)]
                fit_MA=fit_const_MA; tfmins,tcmins=tfmins_const,tcmins_const
                if fit_MA is not None:
                    pars_jk,probs_jk,fitlabels=fit_MA
                    tfs=removeDuplicates([fitlabel[0] for fitlabel in fitlabels])
                    mean,err=jackme(pars_jk[:,0])
                    plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                    ax.legend()
                
                for fit in fits:
                    (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
                    if tfmin not in tfmins or tcmin not in tcmins:
                        continue
                    itcmin=tcmins.index(tcmin)
                    mfc=None
                    if fit_MA is not None and (tfmin,tcmin) in fitlabels:
                        mfc='white'
                    mean,err=jackme(pars_jk[:,0])
                    plt_x=(tfmin+itcmin*0.1)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_fit[itcmin],mfc=mfc)

            show='fit_sum'
            fits=fits_sum
            if show in shows and fits is not None:
                ax=axs[irow,shows.index(show)]
                fit_MA=fit_sum_MA; tfmins,tcmins=tfmins_sum,tcmins_sum
                if fit_MA is not None:
                    pars_jk,probs_jk,fitlabels=fit_MA
                    tfs=removeDuplicates([fitlabel[0] for fitlabel in fitlabels])
                    mean,err=jackme(pars_jk[:,0])
                    plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                    ax.legend()
                
                for fit in fits:
                    (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
                    if tfmin not in tfmins or tcmin not in tcmins:
                        continue
                    itcmin=tcmins.index(tcmin)
                    mfc=None
                    if fit_MA is not None and (tfmin,tcmin) in fitlabels:
                        mfc='white'
                    mean,err=jackme(pars_jk[:,0])
                    plt_x=(tfmin+itcmin*0.1)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_fit[itcmin],mfc=mfc)
                
            show='fit_2st'
            fits=fits_2st
            if show in shows and fits is not None:
                ax=axs[irow,shows.index(show)]
                fit_MA=fit_2st_MA; tfmins,tcmins=tfmins_2st,tcmins_2st
                if fit_MA is not None:
                    pars_jk,probs_jk,fitlabels=fit_MA
                    tfs=removeDuplicates([fitlabel[0] for fitlabel in fitlabels])
                    mean,err=jackme(pars_jk[:,0])
                    plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                    ax.legend()
                
                for fit in fits:
                    (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
                    if tfmin not in tfmins or tcmin not in tcmins:
                        continue
                    itcmin=tcmins.index(tcmin)
                    mfc=None
                    if fit_MA is not None and (tfmin,tcmin) in fitlabels:
                        mfc='white'
                    mean,err=jackme(pars_jk[:,0])
                    plt_x=(tfmin+itcmin*0.1)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_fit[itcmin],mfc=mfc)
            
    def makePlot_3pt_ChristosStyle(list_tf2ratio_fits,xunits=1,yunits=1,tcmin_rainbow=1,selection=None):
        if type(xunits)!=list:
            xunits=[xunits]*len(list_tf2ratio_fits)
        if type(yunits)!=list:
            yunits=[yunits]*len(list_tf2ratio_fits)
        
        fig, axs = getFigAxs(len(list_tf2ratio_fits),4,Lrow=4,Lcol=6,sharex='col',sharey='row', gridspec_kw={'width_ratios': [3, 2, 2, 2]})
        irow=-1
        axs[irow,0].set_xlabel(r'$t_{\rm ins}-t_{s}/2$ [fm]')        
        axs[irow,1].set_xlabel(r'$t_{s}$ [fm]')
        axs[irow,2].set_xlabel(r'$t_{s}^{\rm low}$ [fm]')
        axs[irow,3].set_xlabel(r'Fit Prob.')
        
        ax=axs[irow,3]
        xticks=[1,3,10,30,90]
        ax.set_xlim([xticks[0]/2,150])
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}%' for x in xticks])
        
        def get_fittype(tf2ratio_fits):
            tf2ratio,fits=tf2ratio_fits[:2]
            fitlabel,pars_jk,chi2_jk,Ndof=fits[0]
            if pars_jk.shape[1]==1:
                fittype='1st'
            elif len(list_tf2ratio_fits[irow])==3:
                fittype='2st2step'
            elif pars_jk.shape[1]==2:
                fittype='sum'
            else:
                fittype='2st'
            return fittype
        
        tfmin_ratio=1e10; tfmax_ratio=0
        tfmin_fit=1e10; tfmax_fit=0
        for irow in range(len(list_tf2ratio_fits)):
            xunit=xunits[irow]; yunit=yunits[irow]
            tf2ratio,fits=list_tf2ratio_fits[irow][:2]
            fittype=get_fittype(list_tf2ratio_fits[irow])
                
            pars_jk,probs_jk=jackMA(fits)
            mean,err=jackme(pars_jk)
            plt_y_quote,plt_yerr_quote=mean[0]*yunit,err[0]*yunit

            tfs=list(tf2ratio.keys()); tfs.sort()
            tfs_rainbow=[tf for tf in tfs if tf%2==0 and tf>=2*tcmin_rainbow and jackme(tf2ratio[tf][:,tf//2])[1]*yunit<plt_yerr_quote*5]
            if len(tfs_rainbow)==0:
                tfs_rainbow=[tf for tf in tfs if tf%2==0 and tf>=2*tcmin_rainbow and jackme(tf2ratio[tf][:,tf//2])[1]*yunit<plt_yerr_quote*20]
            # print(irow,tfs_rainbow)
            
            tcmins=list(set([fit[0][1] for fit in fits])); tcmins.sort()
            tfmins=list(set([fit[0][0] for fit in fits])); tfmins.sort()
            
            tfmin_ratio=min(tfmin_ratio,np.min(tfs_rainbow))
            tfmax_ratio=max(tfmax_ratio,np.max(tfs_rainbow))
            
            tfmin_fit=min(tfmin_fit,np.min(tfmins))
            tfmax_fit=max(tfmax_fit,np.max(tfmins))
            
        axs[0,0].set_xlim(np.array([-tfmax_ratio/2+(tcmin_rainbow-1),tfmax_ratio/2-(tcmin_rainbow-1)])*xunit)
        axs[0,1].set_xlim(np.array([tfmin_ratio-1,tfmax_ratio+3])*xunit)
        axs[0,2].set_xlim(np.array([tfmin_fit-1,tfmax_fit+1])*xunit)
                
        result=[]
        for irow in range(len(list_tf2ratio_fits)):
            xunit=xunits[irow]; yunit=yunits[irow]
            tf2ratio,fits=list_tf2ratio_fits[irow][:2]
            fittype=get_fittype(list_tf2ratio_fits[irow])
            
            pars_jk,probs_jk=jackMA(fits)
            result.append(pars_jk)
            mean,err=jackme(pars_jk)
            plt_y_quote,plt_yerr_quote=mean[0]*yunit,err[0]*yunit
            for icol in [0,1,2,3]:
                ax=axs[irow,icol]
                plt_x=ax.get_xlim(); plt_y=plt_y_quote; plt_yerr=plt_yerr_quote
                ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='red',alpha=0.2,label=un2str(plt_y,plt_yerr))  
            if fittype in ['1st']:
                axs[irow,0].set_ylim([plt_y_quote-plt_yerr_quote*16,plt_y_quote+plt_yerr_quote*16])
                
            tfs=list(tf2ratio.keys()); tfs.sort()
            tfs_rainbow=[tf for tf in tfs if tf%2==0 and tf>=2*tcmin_rainbow and jackme(tf2ratio[tf][:,tf//2])[1]*yunit<plt_yerr_quote*5]
                
            ax=axs[irow,0]
            for itf,tf in enumerate(tfs_rainbow):
                mean,err=jackme(tf2ratio[tf])
                tcs=np.arange(tcmin_rainbow,tf-tcmin_rainbow+1)
                plt_x=(tcs-tf/2+0.05*(itf-len(tfs_rainbow)/2))*xunit; plt_y=mean[tcs]*yunit; plt_yerr=err[tcs]*yunit
                ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[itf],fmt=fmts16[itf])
                
            ax=axs[irow,1]
            for itf,tf in enumerate(tfs_rainbow):
                if fittype in ['1st','sum']:
                    y_jk=tf2ratio[tf][:,tcmin_rainbow:tf-tcmin_rainbow+1]
                    if np.all(y_jk[:,0]==y_jk[:,-1]): # check if symmetrized
                        y_jk=y_jk[:,:((y_jk.shape[1]+1)//2)]
                    pars_jk,_,_=doFit_const(y_jk)
                    mean,err=jackme(pars_jk)
                elif fittype in ['2st','2st2step']:
                    mean,err=jackme(tf2ratio[tf][:,tf//2])
                plt_x=(tf)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[itf],fmt=fmts16[itf])
                
                if fittype in ['1st','sum'] and tf+1 in tf2ratio.keys():
                    y_jk=tf2ratio[tf+1][:,tcmin_rainbow:tf-tcmin_rainbow+2]
                    if np.all(y_jk[:,0]==y_jk[:,-1]): # check if symmetrized
                        y_jk=y_jk[:,:((y_jk.shape[1]+1)//2)]
                    pars_jk,_,_=doFit_const(y_jk)
                    mean,err=jackme(pars_jk)
                    plt_x=(tf+1)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[itf+len(tfs_rainbow)],fmt=fmts16[itf+len(tfs_rainbow)])
                    
                
            ax=axs[irow,2]
            tcmins=list(set([fit[0][1] for fit in fits])); tcmins.sort()
            tfmins=list(set([fit[0][0] for fit in fits])); tfmins.sort()
            probs=np.mean(probs_jk,axis=0); inds=np.argsort(probs)
            ind_mpf_global=inds[-1]; fitlabel_mpf=fits[ind_mpf_global][0]
            # ind_mpf2_global=inds[-2]; fitlabel_mpf2=fits[ind_mpf2_global][0]
            for i_tfmin,tfmin in enumerate(tfmins):
                tfits=[fit for fit in fits if fit[0][0]==tfmin]
                pars_jk,probs_jk=jackMA(tfits)
                ind_mpf=np.argmax(np.mean(probs_jk,axis=0))
                
                (tfmin,tcmin),pars_jk,chi2_jk,Ndof=tfits[ind_mpf]
                mean,err=jackme(pars_jk)
                
                plt_x=(tfmin)*xunit; plt_y=mean[0]*yunit; plt_yerr=err[0]*yunit
                ax.errorbar(plt_x,plt_y,plt_yerr,color='r',mfc='white' if (tfmin,tcmin)==fitlabel_mpf else None) 
            
            if fittype in ['2st','2st2step']:
                pars_jk,probs_jk=jackMA(fits)
                ind_mpf=np.argmax(np.mean(probs_jk,axis=0))
                (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fits[ind_mpf]
                ax=axs[irow,0]
                t_cut=tcmin
                for itf,tf in enumerate(tfs_rainbow):
                    if tf<tfmin:
                        continue
                    tcs=np.arange(t_cut,tf-t_cut,0.1)
                    if fittype in ['2st2step']:
                        pars_jk_meff2st=list_tf2ratio_fits[irow][2]
                        t=np.array([func_ratio_2st(tf,tcs,pars[0],pars_2pt[1],pars_2pt[2],pars[1],pars[2]) for pars, pars_2pt in zip(pars_jk,pars_jk_meff2st)])
                    else:
                        t=np.array([func_ratio_2st(tf,tcs,*pars) for pars in pars_jk])
                    mean,err=jackme(t)
                    plt_x=(tcs-tf//2)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=colors16[itf],alpha=0.2)   
                ax=axs[irow,1]
                tfs=np.arange(ax.get_xlim()[0]/xunit,ax.get_xlim()[1]/xunit,0.1)
                if fittype in ['2st2step']:
                    pars_jk_meff2st=list_tf2ratio_fits[irow][2]
                    t=np.array([func_ratio_2st(tfs,tfs/2,pars[0],pars_2pt[1],pars_2pt[2],pars[1],pars[2]) for pars, pars_2pt in zip(pars_jk,pars_jk_meff2st)])
                else:
                    t=np.array([func_ratio_2st(tfs,tfs/2,*pars) for pars in pars_jk])
                mean,err=jackme(t)
                plt_x=tfs*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='grey',alpha=0.2)   
                
            ax=axs[irow,3]
            pars_jk,probs_jk=jackMA(fits)
            for ifit, fit in enumerate(fits):
                (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
                prob=np.mean(probs_jk,axis=0)[ifit]
                if prob<1/100:
                    continue
                mean,err=jackme(pars_jk)
                plt_x=(prob)*100; plt_y=mean[0]*yunit; plt_yerr=err[0]*yunit
                ax.errorbar(plt_x,plt_y,plt_yerr,color='r',mfc='white' if (tfmin,tcmin)==fitlabel_mpf else None,
                            label=f'{int(probs[ind_mpf_global]*100)}%; {(tfmin,tcmin)}' if (tfmin,tcmin)==fitlabel_mpf else None)
            ax.legend(fontsize=16)
            
        return fig,axs,result
    
    # def makePlot_3pt_KyliakosStyle_extend(list_dic):
    #     result=[]
    #     fig, axs = getFigAxs(len(list_dic),5,Lrow=4,Lcol=6,sharex='col',sharey='row', gridspec_kw={'width_ratios': [3, 2, 2, 2, 2]})
    #     addColHeader(axs,['rainbow',r'fit to each $t_s$','sum','const','2st'])
    #     irow=-1
    #     axs[irow,0].set_xlabel(r'$t_{\rm ins}-t_{s}/2$ [fm]')        
    #     axs[irow,1].set_xlabel(r'$t_{s}$ [fm]')
    #     axs[irow,2].set_xlabel(r'$t_{s}^{\rm low}$ [fm]')
    #     axs[irow,3].set_xlabel(r'$t_{s}^{\rm low}$ [fm]')

    #     for irow in range(len(list_dic)):
    #         key2res={}
            
    #         dic=list_dic[irow]
    #         def setParameter(default,key):
    #             return dic[key] if key in dic else default
            
    #         xunit=setParameter(1,'xunit')
    #         yunit=setParameter(1,'yunit')
    #         tcmin_rainbow=setParameter(1,'tcmin_rainbow')
    #         tcmin_fit=setParameter(1,'tcmin_fit')

    #         tf2ratio=dic['tf2ratio']
    #         tfs_rainbow=dic['tfs_rainbow']
    #         tfs_bandfit=dic['tfs_bandfit']
    #         tfs_sum=dic['tfs_sum']
    #         tfmins_sum=dic['tfmins_sum']
    #         tfs_WA=dic['tfs_WA']
            
    #         ax=axs[irow,0]
    #         for itf,tf in enumerate(tfs_rainbow):
    #             mean,err=jackme(tf2ratio[tf])
    #             tcs=np.arange(tcmin_rainbow,tf-tcmin_rainbow+1)
    #             plt_x=(tcs-tf/2+0.05*(itf-len(tfs_rainbow)/2))*xunit; plt_y=mean[tcs]*yunit; plt_yerr=err[tcs]*yunit
    #             ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[itf],fmt=fmts16[itf])
                
    #         ax=axs[irow,1]
    #         tcmin=tcmin_fit
    #         pars_WA=[]
            
    #         for i_tf,tf in enumerate(tfs_bandfit):
    #             y_jk=tf2ratio[tf][:,tcmin:tf-tcmin+1]
    #             pars_jk,chi2_jk,Ndof=doFit_const(y_jk,corrQ=False)
                
    #             if tf in tfs_WA:
    #                 pars_WA.append(pars_jk[:,0])
                
    #             mean,err=jackme(pars_jk)
    #             plt_x=tf*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
    #             ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[i_tf],mfc='white' if tf in tfs_WA else None)
            
    #         y_jk=np.transpose(pars_WA)
    #         pars_jk,chi2_jk,Ndof=doFit_const(y_jk,corrQ=False)
    #         key2res['bandfit_WA']=pars_jk[:,0]
    #         mean_wa,err_wa=jackme(pars_jk)
                
    #         ax=axs[irow,2]
    #         tfmins=tfmins_sum
    #         tcmins=[tcmin]
    #         fits=doFit_3pt_sum(tf2ratio,tfmins,tcmins)
    #         for fit in fits:
    #             (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
    #             mean,err=jackme(pars_jk[:,0])
    #             plt_x=tfmin*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
    #             ax.errorbar(plt_x,plt_y,plt_yerr,color='g')  
    #         key2res['sum_first']=fits[0][1][:,0]
            
    #         plt_y=mean_wa[0]*yunit; plt_yerr=err_wa[0]*yunit
    #         axs[irow,0].set_ylim([plt_y-plt_yerr*16,plt_y+plt_yerr*16])
    #         for i in [0,1]:
    #             axs[irow,i].set_xlim(axs[irow,i].get_xlim())
    #             axs[irow,i].fill_between(axs[irow,i].get_xlim(),plt_y-plt_yerr,plt_y+plt_yerr,color='grey',alpha=0.2,label=un2str(plt_y,plt_yerr))  
    #         axs[irow,1].legend() 
            
    #         pars_jk,probs_jk=jackMA(fits)
    #         key2res['sum_MA']=pars_jk[:,0]
    #         mean,err=jackme(pars_jk[:,0])
    #         plt_y=mean*yunit; plt_yerr=err*yunit
    #         axs[irow,2].fill_between(axs[irow,2].get_xlim(),plt_y-plt_yerr,plt_y+plt_yerr,color='green',alpha=0.2,label=un2str(plt_y,plt_yerr))  
    #         axs[irow,2].legend() 
            

            
    #         ax=axs[irow,3]
    #         tcmins_color=[2,3,4,5,6,7,8,9]
    #         tfmins=dic['tfmins_const']
    #         tcmins=dic['tcmins_const']
    #         fits=doFit_3pt_1st(tf2ratio,tfmins,tcmins,symmetrizeQ=True)
    #         for fit in fits:
    #             (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
    #             mean,err=jackme(pars_jk[:,0])
    #             plt_x=(tfmin+tcmin*0.1)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
    #             ax.errorbar(plt_x,plt_y,plt_yerr,color=colors8[tcmins_color.index(tcmin)])  
    #         pars_jk,probs_jk=jackMA(fits)
    #         key2res['const_MA']=pars_jk[:,0]
    #         mean,err=jackme(pars_jk[:,0])
    #         plt_y=mean*yunit; plt_yerr=err*yunit
    #         axs[irow,3].set_xlim(axs[irow,3].get_xlim())
    #         axs[irow,3].fill_between(axs[irow,3].get_xlim(),plt_y-plt_yerr,plt_y+plt_yerr,color='pink',alpha=0.2,label=un2str(plt_y,plt_yerr))  
    #         axs[irow,3].legend() 
            
    #         result.append(key2res)
    #     return fig,axs,result

    
    # def makePlot_3pt_KyliakosStyle(list_dic):
    #     result=[]
    #     fig, axs = getFigAxs(len(list_dic),3,Lrow=4,Lcol=6,sharex='col',sharey='row', gridspec_kw={'width_ratios': [3, 2, 2]})
    #     irow=-1
    #     axs[irow,0].set_xlabel(r'$t_{\rm ins}-t_{s}/2$ [fm]')        
    #     axs[irow,1].set_xlabel(r'$t_{s}$ [fm]')
    #     axs[irow,2].set_xlabel(r'$t_{s}^{\rm low}$ [fm]')
        
    #     # colors=['b','orange','g','r','purple','brown','magenta','gold','cyan']
    #     for irow in range(len(list_dic)):
    #         dic=list_dic[irow]
    #         def setParameter(default,key):
    #             return dic[key] if key in dic else default
            
    #         xunit=setParameter(1,'xunit')
    #         yunit=setParameter(1,'yunit')
    #         tcmin_rainbow=setParameter(1,'tcmin_rainbow')
    #         tcmin_fit=setParameter(1,'tcmin_fit')

    #         tf2ratio=dic['tf2ratio']
    #         tfs_rainbow=dic['tfs_rainbow']
    #         tfs_bandfit=dic['tfs_bandfit']
    #         tfs_sum=dic['tfs_sum']
    #         tfmins_sum=dic['tfmins_sum']
    #         tfs_wa=dic['tfs_wa']
            
    #         ax=axs[irow,0]
    #         for itf,tf in enumerate(tfs_rainbow):
    #             mean,err=jackme(tf2ratio[tf])
    #             tcs=np.arange(tcmin_rainbow,tf-tcmin_rainbow+1)
    #             plt_x=(tcs-tf/2+0.05*(itf-len(tfs_rainbow)/2))*xunit; plt_y=mean[tcs]*yunit; plt_yerr=err[tcs]*yunit
    #             ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[itf],fmt=fmts16[itf])
                
    #         ax=axs[irow,1]
    #         tcmin=tcmin_fit
    #         pars_WA=[]
            
    #         for i_tf,tf in enumerate(tfs_bandfit):
    #             y_jk=tf2ratio[tf][:,tcmin:tf-tcmin+1]
    #             def fitfunc(pars):
    #                 return list(pars)*(tf-2*tcmin+1)
    #             pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,[0.04])
                
    #             if tf in tfs_wa:
    #                 pars_WA.append(pars_jk[:,0])
                
    #             mean,err=jackme(pars_jk)
    #             plt_x=tf*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
    #             ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[i_tf],mfc='white' if tf in tfs_wa else None) 
                
    #         ax=axs[irow,2]
    #         tfmins=tfmins_sum
    #         ratioSummedDic={tf:np.sum(tf2ratio[tf][:,tcmin:tf-tcmin+1],axis=1) for tf in tfs_sum}
    #         for tfmin in tfmins:
    #             y_jk=np.transpose([ratioSummedDic[tf] for tf in tfs_sum if tf>=tfmin])
    #             num=y_jk.shape[1]
    #             def fitfunc(pars):
    #                 g,c=pars
    #                 return g*np.arange(num)+c
    #             pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,[0.04,1])
                
    #             mean,err=jackme(pars_jk[:,0])
    #             plt_x=tfmin*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
    #             ax.errorbar(plt_x,plt_y,plt_yerr,color='g')  
                
    #         pars_WA=np.transpose(pars_WA)
    #         def fitfunc(pars):
    #             return list(pars)*len(pars_WA[0])
    #         pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,pars_WA,[0.04],mask='uncorrelated')
    #         result.append(pars_jk)
    #         mean_wa,err_wa=jackme(pars_jk)
                
    #         plt_y=mean_wa[0]*yunit; plt_yerr=err_wa[0]*yunit
    #         for i in [0,1,2]:
    #             axs[irow,i].set_xlim(axs[0,i].get_xlim())
    #             axs[irow,i].fill_between(axs[0,i].get_xlim(),plt_y-plt_yerr,plt_y+plt_yerr,color='grey',alpha=0.2,label=un2str(plt_y,plt_yerr))  
    #         axs[irow,1].legend()
        
    #     return fig,axs,result
        
#!============== GEVP ==============#
if True:
    def GEVP(Ct,t0List,tList=None,tvList=None):
        '''
        Ct: indexing from t=0
        t0List>=0: t0=t0List
        t0List<0: t0=t-|t0List|
        tv: reference time for getting wave function (Not return wave function if tv is None) 
        # Return #
        eVecs (the one to combine source operators): return (time,n,i) but (time,i,n) in the middle
        wave function returns if tv is not None
        '''
        Ct=Ct.astype(complex)
        (t_total,N_op,N_op)=Ct.shape
        if tList is None:
            tList=range(t_total)
        tList=np.array(tList)
        if type(t0List)==int:
            if t0List>=0:
                t0List=[t0List for t in tList]
            else:
                t0List=[t+t0List if t+t0List>0 else 1 for t in tList]
        elif type(t0List)==str:
            if t0List=='t/2':
                t0List=[(t+1)//2 for t in tList]
        t0List=[t0 if t!=t0 else 0 if t!=0 else 1 for t,t0 in zip(tList,t0List)] # we would never use t==t0 case, this is meant to avoid some warning msg
        t0List=np.array(t0List)
        Ct0=Ct[t0List]
        choL=np.linalg.cholesky(Ct0) # Ct0=choL@choL.H
        choLInv=np.linalg.inv(choL)
        choLInvDag=np.conj(np.transpose(choLInv,[0,2,1]))
        w_Ct=choLInv@Ct[tList]@choLInvDag
        (eVals,w_eVecs)=np.linalg.eig(w_Ct)
        eVals=np.real(eVals)
        
        for ind,t in enumerate(tList):
            t0=t0List[ind]
            sortList=np.argsort(-eVals[ind]) if t0<t else np.argsort(eVals[ind]) 
            (eVals[ind],w_eVecs[ind])=(eVals[ind][sortList],w_eVecs[ind][:,sortList])

        eVecs=choLInvDag@w_eVecs
        
        if tvList is not None:
            if type(tvList)==str:
                if tvList=='t0':
                    tvList=t0List
                elif tvList=='t':
                    tvList=tList
            tvList=np.array(tvList)
            tmp=np.conj(np.transpose(eVecs,[0,2,1]))@Ct[tvList]@eVecs
            tmp=np.real(tmp[:,range(N_op),range(N_op)])
            powers=np.array([ tv/(t-t0) if t!=t0 else 0 for t,t0,tv in zip(tList,t0List,tvList)])
            fn=np.sqrt( tmp / (eVals**powers[:,None]))
            eVecs_normalized=eVecs/fn[:,None,:]
            eVecs_normalized=np.transpose(eVecs_normalized,[0,2,1]) # v^n_i
            Zin=np.linalg.inv(eVecs_normalized) # (Z)_in
            
            return (eVals,eVecs_normalized,Zin)

        return (eVals,np.transpose(eVecs,[0,2,1]))

#!============== ensemble info ==============#
if True:
    ens2full={'a24':'cA211.53.24','a':'cA2.09.48','b':'cB211.072.64','c':'cC211.060.80','d':'cD211.054.96','e':'cE211.044.112'}
    ens2label={'a24':'A24','a':'A48','b':'B64','c':'C80','d':'D96','e':'E112'}
    ens2a={'a24':0.0908,'a':0.0938,'b':0.07957,'c':0.06821,'d':0.05692,'e':0.04892} # fm
    ens2NL={'a24':24,'a':48,'b':64,'c':80,'d':96,'e':112}
    ens2NT={'a24':24*2,'a':48*2,'b':64*2,'c':80*2,'d':96*2,'e':112*2}
    ens2amul={'b':0.00072,'c':0.00060,'d':0.00054,'e':0.00044}
    
    ens2amul_iso={'b':0.0006669,'c':0.0005864,'d':0.0004934,'e': 0.0004306}
    ens2amul_iso_err={'b':0.0000028,'c':0.0000034,'d':0.0000024,'e': 0.0000023}

    ens2aInv={ens:1/(ens2a[ens]*hbarc) for ens in ens2a.keys()} # MeV

#!============== moments ==============#
if True:
    j2j1={
        'jq':[[1,'j+'],[1,'js'],[1,'jc']],
        'jv1':[[1,'j-']],
        'jv2':[[1,'j+'],[-2,'js']],
        'jv3':[[1,'j+'],[1,'js'],[-3,'jc']]
    }
    fla2iso={
        'u':[(1/4,'q'),(1/2,'v1'),(1/6,'v2'),(1/12,'v3')],
        'd':[(1/4,'q'),(-1/2,'v1'),(1/6,'v2'),(1/12,'v3')],
        's':[(1/4,'q'),(-1/3,'v2'),(1/12,'v3')],
        'c':[(1/4,'q'),(-1/4,'v3')]
    }
    fla2iso_conn={
        'u':[(1/2,'q'),(1/2,'v1')],
        'd':[(1/2,'q'),(-1/2,'v1')],
    }
    
    lat_a2s_plt=np.arange(0,0.009,0.001)
    
    def extendBare_avgx(key2bare):
        keys=list(key2bare)
        enss=removeDuplicates([ens for ens,j in keys])
        stouts=removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
        
        for ens in enss:
            ZERO=key2bare[(ens,'j+;conn')]*0
            key2bare[(ens,'js;conn')]=ZERO
            key2bare[(ens,'jc;conn')]=ZERO
            key2bare[(ens,'j-;disc')]=ZERO
            for cd in ['conn','disc']:
                for j in j2j1.keys():
                    key2bare[(ens,f'{j};{cd}')]=np.sum([factor*key2bare[(ens,f'{j1};{cd}')] for factor,j1 in j2j1[j]],axis=0)
            
    def bareRC2phy_avgx(key2bare,ens2RCs,mn_conn='mu=nu',mn_disc='mu!=nu',mn_g='mu!=nu'):
        keys=list(key2bare)
        enss=removeDuplicates([ens for ens,j in keys])
        stouts=removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
        
        key2phy={}
        for ens in enss:
            for j in ['jv1','jv2','jv3']:
                key2phy[(ens,f'{j};conn')]=ens2RCs[ens][f'Zqq({mn_conn})']*key2bare[(ens,f'{j};conn')]
                key2phy[(ens,f'{j};disc')]=ens2RCs[ens][f'Zqq({mn_disc})']*key2bare[(ens,f'{j};disc')]
                key2phy[(ens,f'{j}')]=key2phy[(ens,f'{j};conn')]+key2phy[(ens,f'{j};disc')]
                
            for stout in stouts:
                key2phy[(ens,f'jq;conn;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_conn})']*key2bare[(ens,'jq;conn')]
                key2phy[(ens,f'jq;disc;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jq;0mix;stout{stout}')]=key2phy[(ens,f'jq;conn;stout{stout}')]+key2phy[(ens,f'jq;disc;stout{stout}')]
                
                key2phy[(ens,f'jq;mix;stout{stout}')]=ens2RCs[ens][f'Zqg({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jq;stout{stout}')]=key2phy[(ens,f'jq;0mix;stout{stout}')]+key2phy[(ens,f'jq;mix;stout{stout}')]
                
                key2phy[(ens,f'jg;0mix;stout{stout}')]=ens2RCs[ens][f'Zgg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jg;mix;stout{stout}')]=ens2RCs[ens][f'Zgq({mn_conn})']*key2bare[(ens,'jq;conn')]+ens2RCs[ens][f'Zgq({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jg;stout{stout}')]=key2phy[(ens,f'jg;mix;stout{stout}')]+key2phy[(ens,f'jg;0mix;stout{stout}')]
                
                key2phy[(ens,f'jtot;stout{stout}')]=key2phy[(ens,f'jq;stout{stout}')]+key2phy[(ens,f'jg;stout{stout}')]
                
                for fla in fla2iso.keys():
                    key2phy[(ens,f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(ens,f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(ens,f'j{iso}')])  for factor,iso in fla2iso[fla]],axis=0)
        
        for j in ['jv1','jv2','jv3']+[f'jq;stout{stout}' for stout in stouts]+[f'jg;stout{stout}' for stout in stouts]:
            ens2dat={ens:key2phy[(ens,j)] for ens in enss}
            fits=doFit_continuumExtrapolation(ens2dat,lat_a2s_plt=lat_a2s_plt)
            for fit in fits:
                fitlabel,pars_jk,chi2_jk,Ndof=fit
                key2phy[(f'a=#_{fitlabel}',j)]=pars_jk
            pars_jk,probs_jk=jackMA(fits)
            key2phy[('a=#_MA',j)]=pars_jk
        
        for stout in stouts:
            for fitlabel in ['const','linear','MA']:
                for fla in fla2iso.keys():
                    key2phy[(f'a=#_{fitlabel}',f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(f'a=#_{fitlabel}',f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(f'a=#_{fitlabel}',f'j{iso}')]) for factor,iso in fla2iso[fla]],axis=0)
                key2phy[(f'a=#_{fitlabel}',f'jtot;stout{stout}')]=key2phy[(f'a=#_{fitlabel}',f'jq;stout{stout}')]+key2phy[(f'a=#_{fitlabel}',f'jg;stout{stout}')]
        
        return key2phy
    
    def bareRC2phy_avgx_pre(key2bare,ens2RCs,ens2RCs_pre,mn_conn='mu=nu',mn_disc='mu!=nu',mn_g='mu!=nu'):
        keys=list(key2bare)
        enss=removeDuplicates([ens for ens,j in keys]); enss=['b']
        stouts=removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
        
        key2phy={}
        for ens in enss:
            for j in ['jv1','jv2','jv3']:
                key2phy[(ens,f'{j};conn')]=ens2RCs[ens][f'Zqq({mn_conn})']*key2bare[(ens,f'{j};conn')]
                key2phy[(ens,f'{j};disc')]=ens2RCs[ens][f'Zqq({mn_disc})']*key2bare[(ens,f'{j};disc')]
                key2phy[(ens,f'{j}')]=key2phy[(ens,f'{j};conn')]+key2phy[(ens,f'{j};disc')]
                
            for stout in stouts:
                key2phy[(ens,f'jq;conn;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_conn})']*key2bare[(ens,'jq;conn')]
                key2phy[(ens,f'jq;disc;stout{stout}')]=ens2RCs_pre[ens][f'Zqq^s^{stout}({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jq;0mix;stout{stout}')]=key2phy[(ens,f'jq;conn;stout{stout}')]+key2phy[(ens,f'jq;disc;stout{stout}')]
                
                key2phy[(ens,f'jq;mix;stout{stout}')]=ens2RCs_pre[ens][f'Zqg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jq;stout{stout}')]=key2phy[(ens,f'jq;0mix;stout{stout}')]+key2phy[(ens,f'jq;mix;stout{stout}')]
                
                key2phy[(ens,f'jg;0mix;stout{stout}')]=ens2RCs_pre[ens][f'Zgg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jg;mix;stout{stout}')]=ens2RCs[ens][f'Zgq({mn_conn})']*key2bare[(ens,'jq;conn')]+ens2RCs_pre[ens][f'Zgq^{stout}({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jg;stout{stout}')]=key2phy[(ens,f'jg;mix;stout{stout}')]+key2phy[(ens,f'jg;0mix;stout{stout}')]
                
                key2phy[(ens,f'jtot;stout{stout}')]=key2phy[(ens,f'jq;stout{stout}')]+key2phy[(ens,f'jg;stout{stout}')]
                
                for fla in fla2iso.keys():
                    key2phy[(ens,f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(ens,f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(ens,f'j{iso}')])  for factor,iso in fla2iso[fla]],axis=0)        
        
        return key2phy

    def convert_key2phy_stout(key2phy_old,stout):
        key2phy={}
        for key in key2phy_old.keys():
            ens,j=key
            if j.endswith(f';stout{stout}'):
                key2phy[(ens,j.removesuffix(f';stout{stout}'))]=key2phy_old[key]
            if 'stout' not in j:
                key2phy[key]=key2phy_old[key]
        return key2phy                
    
    def makePlot_a2dependence_avgx(list_dic):
        Ncol=len(list_dic)
        fig, axs = getFigAxs(2,Ncol,Lrow=4,Lcol=6,sharex=True,sharey='row')
        ax=axs[0,0]
        ax.set_xlim([0,lat_a2s_plt[-1]])
        ax.set_xticks([0,0.003,0.006])
        ax.set_ylim([0.2,1.4])
        ax.set_yticks([0.4,0.6,0.8,1.0,1.2])
        ax.set_ylabel(r'$\langle \mathrm{x} \rangle_{q,g}$')
        ax=axs[1,0]
        ax.set_ylim([-0.1,0.5])
        ax.set_ylabel(r'$\langle \mathrm{x}\rangle_{q}$')
        for icol in range(Ncol):
            axs[0,icol].axhline(1,color='black',ls='--',marker='')
            axs[0,icol].axvline(0,color='black',ls='dotted',marker='')
            axs[1,icol].set_xlabel(r'$a^2$ [fm$^2$]')
            axs[1,icol].axhline(0,color='black',ls='--',marker='')
            
        for icol in range(Ncol):
            dic=list_dic[icol]
            def setParameter(default,key):
                return dic[key] if key in dic else default
            
            key2phy=dic['key2phy']
            key2phy_pre=setParameter(None,'key2phy_pre')
            
            keys=list(key2phy)
            enss=removeDuplicates([ens for ens,j in keys if 'a=#' not in ens])
            
            def get(ens,j):
                return key2phy[(ens,j)]
            def get_pre(ens,j):
                return key2phy_pre[(ens,j)]
            
            j2color={'jq':'purple','jg':'cyan','jtot':'grey','ju':'r','jd':'g','js':'b','jc':'orange'}
            j2label={'jq':'q','jg':'g','jtot':'N','ju':'u','jd':'d','js':'s','jc':'c'}
            j2fmt={'jq':'d','jg':'s','jtot':'o','ju':'^','jd':'v','js':'<','jc':'>'}
            
            
            ax=axs[0,icol]
            js=['jq','jtot','jg']
            for ij,j in enumerate(js):
                color=j2color[j]
                mean,err=jackme(get('a=#_MA',j))
                label=rf'$\langle x\rangle _{{{j2label[j]}}}=$'+un2str(mean[0],err[0],forceResult=1)
                for iens,ens in enumerate(enss):
                    plt_x=ens2a[ens]**2+(ij-len(js)/2)*5e-5; plt_y,plt_yerr=jackme(get(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],label=label if iens==0 else None)
                    
                    if key2phy_pre is None or ens not in ['b']:
                        continue
                    plt_x=ens2a[ens]**2+0.0001; plt_y,plt_yerr=jackme(get_pre(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],mfc='white')
                        

                mean,err=jackme(get('a=#_MA',j))
                x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
                ax.plot(x,mean,color=color,linestyle='--',marker='')
                ax.fill_between(x, ymin, ymax, color=color, alpha=0.1)
            ax.legend(fontsize=10,ncol=2)

            ax=axs[1,icol]
            js=['ju','jd','js','jc']
            for ij,j in enumerate(js):
                color=j2color[j]
                mean,err=jackme(get('a=#_MA',j))
                label=rf'$\langle x\rangle _{{{j2label[j]}}}=$'+un2str(mean[0],err[0],forceResult=1)
                for iens,ens in enumerate(enss):
                    plt_x=ens2a[ens]**2+(ij-len(js)/2)*5e-5; plt_y,plt_yerr=jackme(get(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],label=label if iens==0 else None)
                    if key2phy_pre is None or ens not in ['b']:
                        continue
                    plt_x=ens2a[ens]**2+0.0001; plt_y,plt_yerr=jackme(get_pre(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],mfc='white')

                mean,err=jackme(get('a=#_MA',j))
                x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
                ax.plot(x,mean,color=color,linestyle='--',marker='')
                ax.fill_between(x, ymin, ymax, color=color, alpha=0.1)
            ax.legend(fontsize=10,ncol=2)
        
        return fig,axs
    
#!============== obsolete  ==============#
if False:
    app_init=[['pi0i',{'pib'}],['pi0f',{'pia'}],['j',{'j'}],['P',{'pia','pib'}],\
        ['jPi',{'j','pib'}],['jPf',{'pia','j'}],['PJP',{'pia','j','pib'}]]
    diag_init=[
        [['N','N_bw'],{'N'},\
            [[],['pi0i'],['pi0f'],['P'],['pi0f','pi0i'], ['j'],['jPi'],['j','pi0i'],['jPf'],['pi0f','j']]],
        [['T','T_bw'],{'N','pib'},\
            [[],['pi0f'],['j']]],
        [['B2pt','W2pt','Z2pt','B2pt_bw','W2pt_bw','Z2pt_bw'],{'N','pia','pib'},\
            [[]]],
        [['NJN'],{'N','j'},\
            [[],['pi0i'],['pi0f']]],
        [['B3pt','W3pt','Z3pt'],{'N','j','pib'},\
            [[]]],
        # [['NpiJNpi'],{'N','pia','j','pib'},\
        #     [[]]],
    ]

    diags_all=set(); diags_pi0Loopful=set(); diags_jLoopful=set()

    diag2baps={}; diag2dgtp={} # baps=base+apps; dgtp=diagram type
    for app,dgtp in app_init:
        diag2dgtp[app]=dgtp
    for bases,base_dgtp,appss in diag_init:
        for base in bases:
            if base.endswith('_bw'):
                continue
            for apps in appss:
                diag='-'.join([base]+apps)
                diag2baps[diag]=(base,apps)
                diag2dgtp[diag]=set.union(*([base_dgtp]+[diag2dgtp[app] for app in apps]))
                # if diag2dgtp[diag]=={'N','pia'}:
                #     continue
                # if diag2dgtp[diag]=={'N','pia','j'}:
                #     continue

                diags_all.add(diag)
                if 'pi0i' in apps or 'pi0f' in apps:
                    diags_pi0Loopful.add(diag)
                if 'j' in apps:
                    diags_jLoopful.add(diag)
                
    diags_loopful = diags_pi0Loopful | diags_jLoopful
    diags_loopless = diags_all - diags_loopful
    diags_jLoopless = diags_all - diags_jLoopful
    diags_pi0Loopless = diags_all - diags_pi0Loopful

    diag='P'
    diags_all.add(diag); diags_loopless.add(diag); diags_jLoopless.add(diag); diags_pi0Loopless.add(diag)
    diag='pi0f-pi0i'
    diags_all.add(diag); diags_loopful.add(diag); diags_jLoopless.add(diag)

    def load(path,d=0,nmin=6000):
        print('loading: '+path)
        data_load={}
        with h5py.File(path) as f:
            cfgs=[cfg.decode() for cfg in f['cfgs']]
            Ncfg=len(cfgs); Njk=len(jackknife(np.zeros(Ncfg),d=d,nmin=nmin))
            
            datasets=[]
            def visit_function(name,node):
                if isinstance(node, h5py.Dataset):
                    datasets.append(name)
                    # print(len(datasets),name,end='\r')
            f.visititems(visit_function)
                
            N=len(datasets)
            for i,dataset in enumerate(datasets):
                if 'data' in dataset:
                    data_load[dataset]=jackknife(f[dataset][()],d=d,nmin=nmin)
                else:
                    data_load[dataset]=f[dataset][()]
                print(str(i+1)+'/'+str(N)+': '+dataset,end='                           \r')
            print()

        def op_new(op,fla):
            t=op.split(';')
            t[-1]=fla
            return ';'.join(t)
        gjList=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']
        diags=set([dataset.split('/')[1] for dataset in list(data_load.keys()) if 'diags' in dataset])
        opabsDic={}
        for diag in diags:
            opabsDic[diag]=[opab.decode() for opab in data_load['/'.join(['diags',diag,'opabs'])]]
            
        data={'2pt':{},'3pt':{},'VEV':{},'cfgs':[cfgs,Ncfg,Njk]}
        for dataset in data_load.keys():
            if not (dataset.startswith('diags') and 'data' in dataset):
                continue
            _,diag,_,fla=dataset.split('/')
            opabs=opabsDic[diag]
            
            npt='3pt' if '_deltat_' in dataset else '2pt'
            if npt =='2pt':
                for i,opab in enumerate(opabs):
                    opa,opb=str(opab).split('_')
                    flaa,flab=str(fla).split('_')
                    opa=op_new(opa,flaa); opb=op_new(opb,flab)
                    opab=opa+'_'+opb
                    if opab not in data[npt].keys():
                        data[npt][opab]={}
                    data[npt][opab][diag]=data_load[dataset][:,:,i]
            else:
                for i,opab in enumerate(opabs):
                    opa,opb=str(opab).split('_')
                    flaa,j,flab,_,tf=str(fla).split('_')
                    opa=op_new(opa,flaa); opb=op_new(opb,flab)
                    opab=opa+'_'+opb
                    if opab not in data[npt].keys():
                        data[npt][opab]={}
                    for i_gm,gm in enumerate(gjList):
                        insert='_'.join([gm,j,tf])
                        if insert not in data[npt][opab]:
                            data[npt][opab][insert]={}
                        data[npt][opab][insert][diag]=data_load[dataset][:,:,i,i_gm]   

        data['VEV']['j']={}
        for dataset in data_load.keys():
            if not (dataset.startswith('VEV') and 'data' in dataset):
                continue
            npt='VEV'
            _,diag,_,fla=dataset.split('/')
            if diag=='j':
                for i_gm,gm in enumerate(gjList):
                    insert='_'.join([gm,fla])
                    data[npt][diag][insert]=data_load[dataset][:,i_gm]
            elif diag=='pi0f':
                # print(dataset)
                data[npt][diag]={'sgm':data_load[dataset]}
            
        return data

    def getNpar(op):
        return {'p':1,'n,pi+':2,'p,pi0':2,'12':2}[op.split(';')[-1]]

    def getNpars(opa,opb):
        return (getNpar(opa),getNpar(opb))

    def pt2irrep(pt):
        return {'0,0,0':'G1g','0,0,1':'G1','0,0,-1':'G1','0,1,0':'G1','0,-1,0':'G1','1,0,0':'G1','-1,0,0':'G1'}[pt]
    def getop(pt,l,of):
        occ,fla=of
        return ';'.join(['g',pt,pt2irrep(pt),occ,l,fla])
    def getopab(pt,l,ofa,ofb):
        return getop(pt,l,ofa),getop(pt,l,ofb)
    def getops(pt,l,ofs):
        return [getop(pt,l,of) for of in ofs]    
    def op_getl_sgn(op):
        return {'l1':-1,'l2':1}[op.split(';')[-2]]
    def op_flipl(op):
        t=op.split(';')
        t[-2]={'l1':'l2','l2':'l1'}[t[-2]]
        return ';'.join(t)

    gtCj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':1,
        'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':1,'sgmty':1,'sgmtz':1} # gt G^dag gt = (gtCj) G

    fourCPTstar={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':-1,
            'sgmxy':1,'sgmyz':1,'sgmzx':1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1} # g4CPT G^* g4CPT = (fourCPTstar) G
