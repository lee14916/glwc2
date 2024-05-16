import numpy as np
from math import floor, log10
from scipy.optimize import leastsq
from scipy.linalg import cholesky

def jackknife(dat):
    n=len(dat)
    return np.array([np.mean(np.delete(dat,i,axis=0),axis=0) for i in range(n)])
def jackme(dat_jk):
    n=len(dat_jk)
    dat_mean=np.mean(dat_jk,axis=0)
    dat_err=np.sqrt(np.var(dat_jk,axis=0)*(n-1))
    return (dat_mean,dat_err)
def jackmec(dat_jk):
    n=len(dat_jk)
    dat_mean=np.mean(dat_jk,axis=0)
    dat_cov=np.atleast_2d(np.cov(np.array(dat_jk).T)*(n-1)*(n-1)/n)
    dat_err=np.sqrt(np.diag(dat_cov))
    return (dat_mean,dat_err,dat_cov)
def jackmap(func,dat_jk):
    return np.array([func(dat) for dat in dat_jk])
    
def jackfit(fitfunc,y_jk,pars0,estimator=lambda x:[],mask=None):
    y_mean,_,y_cov=jackmec(y_jk)
    if mask is not None:
        if mask is 'uncorrelated':
            y_cov=np.diag(np.diag(y_cov))
        else:
            y_cov=y_cov*mask
        
    cho_L_Inv = np.linalg.inv(cholesky(y_cov, lower=True)) # y_cov^{-1}=cho_L_Inv^T@cho_L_Inv
    fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(pars)-y_mean)
    pars_mean,pars_cov=leastsq(fitfunc_wrapper,pars0,full_output=True)[:2]
    
    pars2obs=lambda pars:np.hstack([estimator(pars),pars])
    def func(dat):
        fitfunc_wrapper2=lambda pars: cho_L_Inv@(fitfunc(pars)-dat)
        pars=leastsq(fitfunc_wrapper2,pars_mean)[0]
        obs=pars2obs(pars)
        return obs
    obs_jk=jackmap(func,y_jk)
        
    pars_jk_mean=obs_jk[:,-len(pars0):].mean(axis=0)
    chi2=np.sum(fitfunc_wrapper(pars_jk_mean)**2)
    Ndata=len(y_mean); Npar=len(pars0); Ndof=Ndata-Npar
    chi2R=chi2/Ndof

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
