import numpy as np
from math import floor, log10
from scipy.optimize import leastsq
from scipy.linalg import cholesky

flag_fast=False

def propagateError(func,mean,cov):
    '''
    y=func(x)=func(mean)+A(x-mean); x~(mean,cov)
    Linear propagation of uncertainty
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
    return (np.array(y_mean),y_cov)

def jackknife(dat_ens,d=1):
    n=len(dat_ens)
    return np.array([np.mean(np.delete(dat_ens,i,axis=0),axis=0) for i in range(n)])
def jackME(dat_jk):
    n=len(dat_jk)
    dat_mean=np.mean(dat_jk,axis=0)
    dat_err=np.sqrt(np.var(dat_jk,axis=0)*(n-1))
    return (dat_mean,dat_err)
def jackMEC(dat_jk):
    n=len(dat_jk)
    dat_mean=np.mean(dat_jk,axis=0)
    dat_cov=np.atleast_2d(np.cov(np.array(dat_jk).T)*(n-1)*(n-1)/n)
    dat_err=np.sqrt(np.diag(dat_cov))
    return (dat_mean,dat_err,dat_cov)
def jackknife_pseudo(mean,cov,n):
    dat_ens=np.random.multivariate_normal(mean,cov*n,n)
    dat_jk=jackknife(dat_ens)
    # do transformation [obs_jk -> A obs_jk + B] to force pseudo mean and err exactly the same
    mean1,_,cov1=jackMEC(dat_jk)
    A=np.sqrt(np.diag(cov)/np.diag(cov1))
    B=mean-A*mean1
    dat_jk=A[None,:]*dat_jk+B[None,:]
    return dat_jk
    
def jackfit(fitfunc,y_jk,pars0,estimator=lambda x:[],correlatedQ=True):
    y_mean,_,y_cov=jackMEC(y_jk)
    if not correlatedQ:
        y_cov=np.diag(np.diag(y_cov))
        
    cho_L_Inv = np.linalg.inv(cholesky(y_cov, lower=True))
    fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(pars)-y_mean)
    pars_mean,pars_cov=leastsq(fitfunc_wrapper,pars0,full_output=True)[:2]

    chi2=np.sum(fitfunc_wrapper(pars_mean)**2)
    Ndata=len(y_mean); Npar=len(pars0); Ndof=Ndata-Npar
    chi2R=chi2/Ndof
    
    pars2obs=lambda pars:np.hstack([estimator(pars),pars])
    if flag_fast: # Generate pseudo jackknife resamples from the single fit 
        obs_mean,obs_cov=propagateError(pars2obs,pars_mean,pars_cov)
        n=len(y_jk)
        obs_jk=jackknife_pseudo(obs_mean,obs_cov,n)
    else:
        def func(dat):
            fitfunc_wrapper2=lambda pars: cho_L_Inv@(fitfunc(pars)-dat)
            pars=leastsq(fitfunc_wrapper2,pars_mean)[0]
            obs=pars2obs(pars)
            return obs
        obs_jk=np.array([func(dat) for dat in y_jk])

    return obs_jk,chi2R,Ndof

def modelAvg(fits):
    '''
    fits=[fit]; fit=(obs_mean,obs_err,chi2R,Ndof)
    '''
    weights=np.exp([-(chi2R*Ndof)/2+Ndof for obs_mean,obs_err,chi2R,Ndof in fits])
    probs=weights/np.sum(weights)
    obs_mean_MA=np.sum(np.array([obs_mean for obs_mean,obs_err,chi2R,Ndof in fits])*probs[:,None],axis=0)
    obs_err_MA=np.sqrt(np.sum(np.array([obs_err**2+obs_mean**2 for obs_mean,obs_err,chi2R,Ndof in fits])*probs[:,None],axis=0)-obs_mean_MA**2)
    return (obs_mean_MA,obs_err_MA,probs)


# uncertainty to string: taken from https://stackoverflow.com/questions/6671053/python-pretty-print-errorbars
def un2str(x, xe, precision=2):
    """pretty print nominal value and uncertainty

    x  - nominal value
    xe - uncertainty
    precision - number of significant digits in uncertainty

    returns shortest string representation of `x +- xe` either as
        x.xx(ee)e+xx
    or as
        xxx.xx(ee)"""
    # base 10 exponents
    x_exp = int(floor(log10(x)))
    xe_exp = int(floor(log10(xe)))

    # uncertainty
    un_exp = xe_exp-precision+1
    un_int = round(xe*10**(-un_exp))

    # nominal value
    no_exp = un_exp
    no_int = round(x*10**(-no_exp))

    # format - nom(unc)exp
    fieldw = x_exp - no_exp
    fmt = '%%.%df' % fieldw
    result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)

    # format - nom(unc)
    fieldw = max(0, -no_exp)
    fmt = '%%.%df' % fieldw
    result2 = (fmt + '(%.0f)') % (no_int*10**no_exp, un_int*10**max(0, un_exp))

    # return shortest representation
    if len(result2) <= len(result1):
        return result2
    else:
        return result1
