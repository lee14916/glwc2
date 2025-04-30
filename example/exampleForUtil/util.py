import os,h5py,warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math,cmath
from math import floor, log10
import pickle
from scipy.optimize import leastsq, curve_fit, fsolve
from scipy.linalg import solve_triangular,cholesky
from inspect import signature

flag_fast=False

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
def jackfit(fitfunc,y_jk,pars0,mask=None):
    y_mean,_,y_cov=jackmec(y_jk)
    if mask is not None:
        if mask == 'uncorrelated':
            y_cov=np.diag(np.diag(y_cov))
        else:
            y_cov=y_cov*mask
        
    cho_L_Inv = np.linalg.inv(cholesky(y_cov, lower=True)) # y_cov^{-1}=cho_L_Inv^T@cho_L_Inv
    fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(pars)-y_mean)
    pars_mean,pars_cov=leastsq(fitfunc_wrapper,pars0,full_output=True)[:2]
    
    def func(dat):
        fitfunc_wrapper2=lambda pars: cho_L_Inv@(fitfunc(pars)-dat)
        pars=leastsq(fitfunc_wrapper2,pars_mean)[0]
        return pars
    pars_jk=jackmap(func,y_jk)
    
    chi2_jk=np.array([[np.sum(fitfunc_wrapper(pars)**2)] for pars in pars_jk])
    Ndata=len(y_mean); Npar=len(pars0); Ndof=Ndata-Npar
    return pars_jk,chi2_jk,Ndof

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

# 
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
    fmt = '%%.%df' % fieldw
    result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)

    # format - nom(unc)
    fieldw = max(0, -no_exp)
    fmt = '%%.%df' % fieldw
    result2 = (fmt + '(%.0f)') % (no_int*10**no_exp, un_int*10**max(0, un_exp))
    if un_exp<0 and un_int*10**un_exp>=1:
        fmt2= '(%%.%df)' % (-un_exp)
        result2 = (fmt + fmt2) % (no_int*10**no_exp, un_int*10**un_exp)

    # return shortest representation
    if len(result2) <= len(result1):
        return result2
    else:
        return result1

# Plot

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