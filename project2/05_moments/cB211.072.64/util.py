import os,h5py,warnings
import numpy as np
np.seterr(invalid=['ignore','warn'][0])
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import matplotlib as mpl
import math,cmath
from math import floor, log10
import pickle
from scipy.optimize import leastsq, curve_fit, fsolve
from scipy.linalg import solve_triangular,cholesky
from inspect import signature
from matplotlib.backends.backend_pdf import PdfPages
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
mpl.rcParams['legend.fontsize'] = 24
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

__all__ = ['np','os','plt','pickle','h5py','mpl','PdfPages']

flag_fast=False # If True, certain functions will be speeded up using approximations.

#!============== small functions ==============#
if True:
    deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)
    npRound=lambda dat,n:np.round(np.array(dat).astype(float),n)
    
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
    
    def moms2dic(moms):
        dic={}
        for i,mom in enumerate(moms):
            dic[tuple(mom)]=i
        return dic
    
    def moms2list(moms):
        return [list(mom) for mom in moms]
    
    def decodeList(l):
        return [ele.decode() for ele in l]
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
    def jackfit(fitfunc,y_jk,pars0,mask=None,parsExtra_jk=None,priors=[],**kargs):
        '''
        priors=[(ind of par, mean, width)]
        '''
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
        return pars_jk,chi2_jk,Ndof
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
        props_jk=np.transpose([weights_jk[:,0]/weightsSum_jk[:,0] for _,weights_jk in temp])
        return pars_jk,props_jk

    def modelAvg(fits):
        '''
        fits=[fit]; fit=(pars_mean,pars_err,chi2,Ndof)
        '''
        weights=np.exp([-chi2/2+Ndof for pars_mean,pars_err,chi2,Ndof in fits])
        props=weights/np.sum(weights)
        pars_mean_MA=np.sum(np.array([pars_mean for pars_mean,pars_err,chi2,Ndof in fits])*props[:,None],axis=0)
        pars_err_MA=np.sqrt(np.sum(np.array([pars_err**2+pars_mean**2 for pars_mean,pars_err,chi2,Ndof in fits])*props[:,None],axis=0)-pars_mean_MA**2)
        return (pars_mean_MA,pars_err_MA,props)
    # def modelAvg(fits): # test
    #     '''
    #     fits=[fit]; fit=(pars_mean,pars_err,chi2,Ndof)
    #     '''
    #     weights=np.exp([-chi2/2+Ndof for pars_mean,pars_err,chi2,Ndof in fits])
    #     props=weights/np.sum(weights)
    #     pars_mean_MA=np.sum(np.array([pars_mean for pars_mean,pars_err,chi2,Ndof in fits])*props[:,None],axis=0)
    #     pars_err_MA=np.sqrt(np.sum(np.array([pars_err**2+pars_mean**2 for pars_mean,pars_err,chi2,Ndof in fits])*props[:,None],axis=0)-pars_mean_MA**2)
    #     res=0
    #     for i,prop in enumerate(props):
    #         # if prop<0.0001:
    #         #     continue
    #         pars_mean,pars_err,_,_=fits[i]
    #         tmean=pars_mean[0]; terr=pars_err[0]
    #         tmean_MA=pars_mean_MA[0]
    #         res=res+(terr**2+(tmean-tmean_MA)**2)*prop
    #         # print((terr**2+(tmean-tmean_MA)**2)*prop)
    #         # print(i,"%0.2f" %prop,tmean,terr,terr**2*prop/(pars_err_MA[0]**2),(tmean-tmean_MA)**2*prop/(pars_err_MA[0]**2))
    #         print(i,"%0.2f" %prop,"%0.2f" %(terr**2*prop/(pars_err_MA[0]**2)),"%0.2f" %((tmean-tmean_MA)**2*prop/(pars_err_MA[0]**2)))
    #     print(np.sqrt(res),pars_err_MA[0])
    #     print()
    #     # pars_err_MA=np.sqrt(np.sum(np.array([pars_err**2 for pars_mean,pars_err,chi2,Ndof in fits])*props[:,None],axis=0))
    #     return (pars_mean_MA,pars_err_MA,props)
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
#!============== fit ==============#
if True:
    def find_fitmax(dat,threshold=0.2):
        mean,err=jackme(dat)
        rela=np.abs(err/mean)
        temp=[(i,rela) for i,rela in enumerate(rela) if rela>0.2 and i!=0]
        fitmax=temp[0][0]-1 if len(temp)!=0 else len(mean)-1
        return fitmax
    
    def doFit_const(y_jk,corrQ=True):
        Ndata=y_jk.shape[1]
        if Ndata==1:
            return y_jk,np.zeros(len(y_jk)),0
        def fitfunc(pars):
            return list(pars)*Ndata
        pars_jk,chi2_jk,Ndof=jackfit(fitfunc,y_jk,[np.mean(y_jk)],mask=None if corrQ else 'uncorrelated')
        return pars_jk,chi2_jk,Ndof

    def doFit_2pt(dat,tmins,func,pars0,downSampling=1,corrQ=True):
        tmax=find_fitmax(dat)
        fits=[]
        for tmin in tmins:
            ts=np.arange(tmin,tmax,downSampling)
            def fitfunc(pars):
                return func(ts,*pars)
            y_jk=dat[:,ts]
            pars_jk,chi2_jk,Ndof=jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
            pars0=np.mean(pars_jk,axis=0)
            fits.append([tmin,pars_jk,chi2_jk,Ndof])
        return fits

    func_c2pt_1st=lambda t,E0,c0: c0*np.exp(-E0*t)
    func_c2pt_2st=lambda t,E0,c0,dE1,rc1: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t))
    func_c2pt_3st=lambda t,E0,c0,dE1,rc1,dE2,rc2: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t) + rc2*np.exp(-dE2*t))
    func_meff_1st=lambda t,E0: np.log(func_c2pt_1st(t,E0,1)/func_c2pt_1st(t+1,E0,1))
    func_meff_2st=lambda t,E0,dE1,rc1: np.log(func_c2pt_2st(t,E0,1,dE1,rc1)/func_c2pt_2st(t+1,E0,1,dE1,rc1))
    func_meff_3st=lambda t,E0,dE1,rc1,dE2,rc2: np.log(func_c2pt_3st(t,E0,1,dE1,rc1,dE2,rc2)/func_c2pt_3st(t+1,E0,1,dE1,rc1,dE2,rc2))
    def doFit_meff_nst(meff,tminss,pars0,downSampling=1,corrQ=True):
        Nst=len(tminss)
        fits_1st=doFit_2pt(meff,tminss[0],func_meff_1st,pars0[:1],downSampling=downSampling,corrQ=corrQ)
        if Nst==1:
            return [fits_1st]
        
        pars_jk,props_jk=jackMA(fits_1st)
        pars0[:1]=np.mean(pars_jk,axis=0)
        fits_2st=doFit_2pt(meff,tminss[1],func_meff_2st,pars0[:3],downSampling=downSampling,corrQ=corrQ)
        if Nst==2:
            return [fits_1st,fits_2st]
        
        pars_jk,props_jk=jackMA(fits_2st)
        pars0[:3]=np.mean(pars_jk,axis=0)
        fits_3st=doFit_2pt(meff,tminss[2],func_meff_3st,pars0[:5],downSampling=downSampling,corrQ=corrQ)
        if Nst==3:
            return [fits_1st,fits_2st,fits_3st]
        
    def doFit_3ptSym_1st(tf2ratio_para,tfmins,tcmins,pars0=None,downSampling=[1,1],symmetrizeQ=False,corrQ=True):
        tf2ratio=tf2ratio_para.copy()
        tfs=list(tf2ratio.keys()); tfs.sort()
        if symmetrizeQ:
            symmetrizeRatio(tf2ratio)
        if pars0 is None:
            tfmax=np.max(tfs)
            g=np.mean(tf2ratio[tfmax][:,tfmax//2])
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
                pars_jk,chi2_jk,Ndof=jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
                fits.append([(tfmin,tcmin),pars_jk,chi2_jk,Ndof])
        return fits
        
    func_c3pt_2st=lambda tf,tc,E0a,E0b,a00,dE1a,dE1b,ra01,ra10,ra11: a00*np.exp(-E0a*(tf-tc))*np.exp(-E0b*tc)*(1 + ra01*np.exp(-dE1b*tc) + ra10*np.exp(-dE1a*(tf-tc)) + ra11*np.exp(-dE1a*(tf-tc))*np.exp(-dE1b*tc)) \
        if a00!=0 else np.exp(-E0a*(tf-tc))*np.exp(-E0b*tc)*(ra01*np.exp(-dE1b*tc) + ra10*np.exp(-dE1a*(tf-tc)) + ra11*np.exp(-dE1a*(tf-tc))*np.exp(-dE1b*tc))
    func_ratio_2st=lambda tf,tc,g,dE1,rc1,ra01,ra11:func_c3pt_2st(tf,tc,0,0,g,dE1,dE1,ra01,ra01,ra11)/func_c2pt_2st(tf,0,1,dE1,rc1)
    def doFit_3ptSym_2st2step(tf2ratio_para,tfmins,tcmins,pars_jk_meff2st,pars0=None,downSampling=[1,1],symmetrizeQ=False,corrQ=True):
        tf2ratio=tf2ratio_para.copy()
        tfs=list(tf2ratio.keys()); tfs.sort()
        if symmetrizeQ:
            for tf in tfs:
                tf2ratio[tf]=(tf2ratio[tf]+tf2ratio[tf][:,::-1])/2
        if pars0 is None:
            tfmax=np.max(tfs)
            g=np.mean(tf2ratio[tfmax][:,tfmax//2])
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
                pars_jk,chi2_jk,Ndof=jackfit(fitfunc,y_jk,pars0,parsExtra_jk=pars_jk_meff2st,mask=None if corrQ else 'uncorrelated')
                fits.append([(tfmin,tcmin),pars_jk,chi2_jk,Ndof])
        return fits
#!============== plot ==============#
if True:
    colors8=['r','g','b','orange','purple','brown','magenta','olive']
    fmts8=['s','o','d','^','v','<','>','*']
    
    colors16=['blue','orange','green','red','purple','brown','pink','grey','olive','cyan','lime','tan','peru','magenta','gold','skyblue']
    fmts16=['o','v','^','<','>','d','X','P','s','h','p','*','H','8','D','.']
    
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

    def finalizePlt(file=None,closeQ=None):
        if closeQ is None:
            closeQ=False if file is None else True
        plt.tight_layout()
        if file!=None:
            plt.savefig(file,bbox_inches="tight")
        if closeQ:
            plt.close()
    
    def makePDF(file,figs):
        pdf = PdfPages(file)
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
            
    def makePlot_2pt_SimoneStyle(meff,fitss,xunit=1,yunit=1,mN_exp=None):
        fig, axd = plt.subplot_mosaic([['f1','f1','f1'],['f2','f2','f3']],figsize=(24,10))
        (ax1,ax2,ax3)=(axd[key] for key in ['f1','f2','f3'])
        ax1.set_xlabel(r'$t$ [fm]')
        ax2.set_xlabel(r'$t_{\mathrm{min}}$ [fm]')
        ax3.set_xlabel(r'$t_{\mathrm{min}}$ [fm]')
        ax1.set_ylabel(r'$m_N^{\mathrm{eff}}$ [GeV]')
        ax2.set_ylabel(r'$m_N$ [GeV]')
        ax3.set_ylabel(r'$E_1$ [GeV]')

        mean,err=jackme(meff)
        fitmax=find_fitmax(meff)
        
        tmin=1; tmax=fitmax+1
        plt_x=np.arange(tmin,tmax)*xunit; plt_y=mean[tmin:tmax]*yunit; plt_yerr=err[tmin:tmax]*yunit
        ax1.errorbar(plt_x,plt_y,plt_yerr,color='black',fmt='s')
        
        if mN_exp is not None:
            ax1.axhline(y=mN_exp,color='black',linestyle = '--', marker='')
            ax2.axhline(y=mN_exp,color='black',linestyle = '--', marker='', label=r'$m_N^{\mathrm{exp}}=$'+'%0.3f'%mN_exp)
        
        Nst=len(fitss)
        propThreshold=0.1
        chi2Size=12
        DNpar=1 # DNpar=1 if meffQ else 0
        percentage_shiftMultiplier=1.5
        
        if Nst==0:
            return fig,axd
        
        color='r'
        fits=fitss[0]    
        fitmins=[fit[0] for fit in fits]
        pars_jk,props_jk=jackMA(fits)
        props_mean=np.mean(props_jk,axis=0)
        ind_mpf=np.argmax(np.mean(props_jk,axis=0))    
        pars_mean,pars_err=jackme(pars_jk)
        plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
        ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2,label=r'$m_N^{\mathrm{1st}}=$'+un2str(plt_y,plt_yerr))
        ax1.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*40])
        ax2.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*30])
        for i,fit in enumerate(fits):
            fitmin,pars_jk,chi2_jk,Ndof=fit; prop=props_mean[i]
            (pars_mean,pars_err)=jackme(pars_jk)
            chi2R=np.mean(chi2_jk)/Ndof
            showQ = i==ind_mpf if propThreshold is None else prop>propThreshold
            
            plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='s',color=color,mfc='white' if showQ else None)
            ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
            ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
            if propThreshold is not None and prop>propThreshold:
                ax2.annotate(f"{int(prop*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
            
        if Nst==1:
            return fig,axd
        
        color='g'
        fits=fitss[1]    
        fitmins=[fit[0] for fit in fits]
        pars_jk,props_jk=jackMA(fits)
        props_mean=np.mean(props_jk,axis=0)
        ind_mpf=np.argmax(np.mean(props_jk,axis=0)) 
        t=pars_jk.copy()
        t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
        pars_mean,pars_err=jackme(t)
        plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
        ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$m_N^{\mathrm{2st}}=$'+un2str(plt_y,plt_yerr))
        plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
        ax3.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_1^{\mathrm{2st}}=$'+un2str(plt_y,plt_yerr))
        ax3.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*30])
        for i,fit in enumerate(fits):
            fitmin,pars_jk,chi2_jk,Ndof=fit; prop=props_mean[i]
            t=pars_jk.copy()
            t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
            (pars_mean,pars_err)=jackme(t)
            chi2R=np.mean(chi2_jk)/Ndof
            showQ = i==ind_mpf if propThreshold is None else prop>propThreshold
            
            plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='o',color=color,mfc='white' if showQ else None)
            ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
            ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
            if propThreshold is not None and prop>propThreshold:
                ax2.annotate(f"{int(prop*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
            
            plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
            ax3.errorbar(plt_x,plt_y,plt_yerr,fmt='o',color=color,mfc='white' if showQ else None)
            ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
            ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
            if propThreshold is not None and prop>propThreshold:
                ax3.annotate(f"{int(prop*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')

        if Nst==2:
            if mN_exp is not None:
                ax2.legend(fontsize=16)
            return fig,axd
        
        color='b'
        fits=fitss[2]    
        fitmins=[fit[0] for fit in fits]
        pars_jk,props_jk=jackMA(fits)
        props_mean=np.mean(props_jk,axis=0)
        ind_mpf=np.argmax(np.mean(props_jk,axis=0))    
        t=pars_jk.copy()
        t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
        pars_mean,pars_err=jackme(t)
        plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
        ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$m_N^{\mathrm{3st}}=$'+un2str(plt_y,plt_yerr))
        plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
        ax3.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_1^{\mathrm{3st}}=$'+un2str(plt_y,plt_yerr))    
        for i,fit in enumerate(fits):
            fitmin,pars_jk,chi2_jk,Ndof=fit; prop=props_mean[i]
            t=pars_jk.copy()
            t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
            (pars_mean,pars_err)=jackme(t)
            chi2R=np.mean(chi2_jk)/Ndof
            showQ = i==ind_mpf if propThreshold is None else prop>propThreshold
            
            plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='d',color=color,mfc='white' if showQ else None)
            ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
            ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
            if propThreshold is not None and prop>propThreshold:
                ax2.annotate(f"{int(prop*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
            
            plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
            ax3.errorbar(plt_x,plt_y,plt_yerr,fmt='d',color=color,mfc='white' if showQ else None)
            ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
            ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center') 
            if propThreshold is not None and prop>propThreshold:
                ax3.annotate(f"{int(prop*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
            
        if Nst==3:
            if mN_exp is not None:
                ax2.legend(fontsize=16)
            return fig,axd      
        
    def makePlot_3pt_ChristosStyle(list_tf2ratio_fits,xunit=1,yunit=1,tcmin_rainbow=1,ylabels=None):
        fig, axs = getFigAxs(len(list_tf2ratio_fits),4,Lrow=4,Lcol=6,sharex='col',sharey='row', gridspec_kw={'width_ratios': [3, 2, 2, 2]})
        irow=-1
        axs[irow,0].set_xlabel(r'$t_{\rm ins}-t_{s}/2$ [fm]')        
        axs[irow,1].set_xlabel(r'$t_{s}^{\rm}$ [fm]')
        axs[irow,2].set_xlabel(r'$t_{s}^{\rm low}$ [fm]')
        axs[irow,3].set_xlabel(r'Fit Prob.')
        
        ax=axs[irow,3]
        xticks=[1,3,10,30]
        ax.set_xlim([xticks[0]/2,xticks[-1]*3])
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}%' for x in xticks])
        
        tfmin_ratio=1e10; tfmax_ratio=0
        tfmin_fit=1e10; tfmax_fit=0
        for irow in range(len(list_tf2ratio_fits)):
            tf2ratio,fits=list_tf2ratio_fits[irow][:2]
            fitlabel,pars_jk,chi2_jk,Ndof=fits[0]
            if pars_jk.shape[1]==1:
                fittype='1st'
            elif len(list_tf2ratio_fits[irow])==3:
                pars_jk_meff2st=list_tf2ratio_fits[irow][2]
                fittype='2st2step'
            else:
                fittype='2st'
                
            pars_jk,props_jk=jackMA(fits)
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
                
        pars_jk_quotes=[]
        for irow in range(len(list_tf2ratio_fits)):
            tf2ratio,fits=list_tf2ratio_fits[irow][:2]
            fitlabel,pars_jk,chi2_jk,Ndof=fits[0]
            if pars_jk.shape[1]==1:
                fittype='1st'
            elif len(list_tf2ratio_fits[irow])==3:
                pars_jk_meff2st=list_tf2ratio_fits[irow][2]
                fittype='2st2step'
            else:
                fittype='2st'
            
            if ylabels is not None:
                ax=axs[irow,0]
                ax.set_ylabel(ylabels[irow])
            
            pars_jk_quote,props_jk=jackMA(fits)
            pars_jk_quotes.append(pars_jk_quote)
            mean,err=jackme(pars_jk_quote)
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
                if fittype in ['1st']:
                    y_jk=tf2ratio[tf][:,tcmin_rainbow:tf-tcmin_rainbow+1]
                    if np.all(y_jk[:,0]==y_jk[:,-1]): # check if symmetrized
                        y_jk=y_jk[:,:((y_jk.shape[1]+1)//2)]
                    pars_jk,_,_=doFit_const(y_jk)
                    mean,err=jackme(pars_jk)
                elif fittype in ['2st','2st2step']:
                    mean,err=jackme(tf2ratio[tf][:,tf//2])
                plt_x=(tf)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                ax.errorbar(plt_x,plt_y,plt_yerr,color=colors16[itf],fmt=fmts16[itf])
                
                if fittype in ['1st'] and tf+1 in tf2ratio.keys():
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
            props=np.mean(props_jk,axis=0); inds=np.argsort(props)
            ind_mpf_global=inds[-1]; ind_mpf2_global=inds[-2]
            fitlabel_mpf=fits[ind_mpf_global][0]
            fitlabel_mpf2=fits[ind_mpf2_global][0]
            for i_tfmin,tfmin in enumerate(tfmins):
                tfits=[fit for fit in fits if fit[0][0]==tfmin]
                pars_jk,props_jk=jackMA(tfits)
                ind_mpf=np.argmax(np.mean(props_jk,axis=0))
                
                (tfmin,tcmin),pars_jk,chi2_jk,Ndof=tfits[ind_mpf]
                mean,err=jackme(pars_jk)
                
                plt_x=(tfmin)*xunit; plt_y=mean[0]*yunit; plt_yerr=err[0]*yunit
                ax.errorbar(plt_x,plt_y,plt_yerr,color='r',mfc='white' if (tfmin,tcmin)==fitlabel_mpf else None) 
            
            if fittype in ['2st','2st2steps']:
                pars_jk,props_jk=jackMA(fits)
                ind_mpf=np.argmax(np.mean(props_jk,axis=0))
                (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fits[ind_mpf]
                ax=axs[irow,0]
                t_cut=tcmin
                for i_tf,tf in enumerate(tfs_rainbow):
                    if tf<tfmin:
                        continue
                    tcs=np.arange(t_cut,tf-t_cut,0.1)
                    if fittype in ['2st2steps']:
                        t=np.array([func_ratio_2st(tf,tcs,pars[0],pars_2pt[1],pars_2pt[2],pars[1],pars[2]) for pars, pars_2pt in zip(pars_jk,pars_jk_meff2st)])
                    else:
                        t=np.array([func_ratio_2st(tf,tcs,*pars) for pars in pars_jk])
                    mean,err=jackme(t)
                    plt_x=(tcs-tf//2)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=colors16[itf],fmt=fmts16[itf],alpha=0.2)   
                ax=axs[irow,1]
                tfs=np.arange(ax.get_xlim()[0]/xunit,ax.get_xlim()[1]/xunit,0.1)
                if fittype in ['2st2steps']:
                    t=np.array([func_ratio_2st(tfs,tfs/2,pars[0],pars_2pt[1],pars_2pt[2],pars[1],pars[2]) for pars, pars_2pt in zip(pars_jk,pars_jk_meff2st)])
                else:
                    t=np.array([func_ratio_2st(tfs,tfs/2,*pars) for pars in pars_jk])
                mean,err=jackme(t)
                plt_x=tfs*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='grey',alpha=0.2)   
                
            ax=axs[irow,3]
            pars_jk,props_jk=jackMA(fits)
            for ifit, fit in enumerate(fits):
                (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
                prop=np.mean(props_jk,axis=0)[ifit]
                if prop<1/100:
                    continue
                mean,err=jackme(pars_jk)
                plt_x=(prop)*100; plt_y=mean[0]*yunit; plt_yerr=err[0]*yunit
                ax.errorbar(plt_x,plt_y,plt_yerr,color='r',mfc='white' if (tfmin,tcmin)==fitlabel_mpf else None,
                            label=f'{int(props[ind_mpf_global]*100)}%; {(tfmin,tcmin)}' if (tfmin,tcmin)==fitlabel_mpf else None)
            ax.legend(fontsize=16)
            
        return fig,axs,pars_jk_quotes

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

    hbarc = 1/197.3
    ens2aInv={ens:1/(ens2a[ens]*hbarc) for ens in ens2a.keys()} # MeV
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
