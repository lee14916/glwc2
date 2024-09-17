import os,h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math,cmath
import pickle
from scipy.optimize import leastsq, curve_fit
from scipy.linalg import solve_triangular,cholesky
from inspect import signature
 
slName='temp'
os.makedirs('aux/sl',exist_ok=True)
if os.path.exists('aux/sl/temp'):
    os.remove('aux/sl/temp')
sldata={}
def sl(key,val="sl_load",flagSave=False):
    if key is None:
        return val
    global sldata
    file='aux/sl/'+slName
    file_backup=file+'_backup'
    if os.path.isfile(file_backup):
        os.replace(file_backup,file)
    if not os.path.isfile(file):
        with open(file,'wb') as f:
            pickle.dump(sldata,f)
    with open(file,'rb') as f:
        sldata=pickle.load(f)
    if not flagSave and key in sldata:
        return sldata[key]
    if val is not "sl_load":
        sldata[key]=val
        os.replace(file,file_backup)
        with open(file,'wb') as f:
            pickle.dump(sldata,f)
        os.remove(file_backup)
    return val

def sl_reset():
    global sldata
    sldata={}
    if os.path.exists('aux/sl/temp'):
        os.remove('aux/sl/temp')
         
flag_fast=False

deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)
npRound=lambda dat,n:np.round(np.array(dat).astype(float),n)

def propagateError(func,mean,cov):
    '''
    y=func(x)=func(mean)+A(x-mean); x~(mean,cov)
    Linear propagation of uncertainty
    Everything are real numbers
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

def extendMeanCov(mean,cov,exts):
    ext_mean=[ext[0] for ext in exts]; ext_err=[ext[1] for ext in exts]
    tmean=np.hstack([mean,ext_mean])
    tcov=np.block([[cov,np.zeros((len(cov),len(ext_mean)))],[np.zeros((len(ext_mean),len(cov))),np.diag(ext_err)**2]])
    return tmean,tcov
    
prefactorDeep=lambda dat,prefactor:np.real(prefactor*dat) if type(dat)==np.ndarray else [prefactorDeep(ele,prefactor) for ele in dat]
meanDeep=lambda dat:np.mean(dat,axis=0) if type(dat)==np.ndarray else [meanDeep(ele) for ele in dat]
def jackknife(in_dat,in_func=lambda dat:np.mean(np.real(dat),axis=0),minNcfg:int=600,d:int=0,outputFlatten=False,sl_key=None,sl_saveQ=False):
    '''
    - in_dat: any-dimensional list of ndarrays. Each ndarray in the list has 0-axis for cfgs
    - in_func: dat -> estimator
    - Estimator: number or 1d-list/array or 2d-list/array or 1d-list of 1d-arrays
    - d: jackknife delete parameter
    ### return: mean,err,cov
    - mean,err: estimator's format reformatted to 1d-list of 1d-arrays
    - cov: 2d-list of 2d-arrays
    '''  
    if sl_key is not None and not sl_saveQ:
        ret=sl(sl_key)
        if ret is not "sl_load":
            return ret
        
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
    (tMean,tCov)=(TnBar, np.atleast_2d(np.cov(Tn1.T)*(n-1)*(n-1)/n))
    # Tn=func(dat); (mean,cov)=(n*Tn-(n-1)*TnBar, np.atleast_2d(np.cov(Tn1.T)*(n-1)*(n-1)/n)) # bias improvement (not suitable for fit)
    tErr=np.sqrt(np.diag(tCov))
    
    if outputFlatten:
        ret=(tMean,tErr,tCov)
        return sl(sl_key,ret,True)
    
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
    return sl(sl_key,ret,True)

def fit0(fitfunc,mean,cov,pars0=None,full_output=False):
    Npar=len(signature(fitfunc).parameters)
    pars0=pars0 if pars0 is not None else [1 for i in range(Npar)]
    cho_L_Inv = np.linalg.inv(cholesky(cov, lower=True))
    y_exp=np.hstack(mean)
    t_fitfunc=lambda pars: cho_L_Inv@(np.hstack(fitfunc(*pars))-y_exp)
    return leastsq(t_fitfunc,pars0,full_output=full_output)

flag_fit_cov2err=False
LEASTSQ_SUCCESS = [1, 2, 3, 4]
def fit(dat,func,fitfunc,estimator=lambda pars:pars,pars0=None,mask_cov=None,jk=True,sl_key=None,sl_saveQ=False):
    '''
    dat: raw data
    func: dat -> y_exp
    fitfunc: pars -> y_model
    return: (est_mean,est_err,est_cov,chi2R,chi2R_err,Ndof,pars) or None if failed
    '''
    if sl_key is not None and not sl_saveQ:
        ret=sl(sl_key)
        if ret is not "sl_load":
            return ret
        
    Npar=len(signature(fitfunc).parameters)
    pars0=pars0 if pars0 is not None else [1 for i in range(Npar)]

    _,_,cov=jackknife(dat,func,outputFlatten=True)
    if flag_fit_cov2err:
        cov=np.diag(np.diag(cov))
    if mask_cov is not None:
        cov=cov*mask_cov
    cho_L_Inv = np.linalg.inv(cholesky(cov, lower=True))
    Ny=len(cov)
    Ndof=Ny-Npar
    
    # 1st fit to all data
    y_exp=np.hstack(func(dat))
    t_fitfunc=lambda pars: cho_L_Inv@(np.hstack(fitfunc(*pars))-y_exp)
    t=leastsq(t_fitfunc,pars0,full_output=True)
    if t[-1] not in LEASTSQ_SUCCESS or t[1] is None:
        ret=None
        return sl(sl_key,ret,True)
    pars,pars_cov=t[:2]
    chi2=np.sum(t_fitfunc(pars)**2)

    if flag_fast or not jk:
        est_mean,est_cov=propagateError(estimator,pars,pars_cov)
        ret=(est_mean,np.sqrt(np.diag(est_cov)),est_cov,chi2/Ndof,None,Ndof,pars)
        return sl(sl_key,ret,True)
    pars0=pars

    def tFunc(dat):
        y_exp=np.hstack(func(dat))
        t_fitfunc=lambda pars: cho_L_Inv@(np.hstack(fitfunc(*pars))-y_exp)
        pars,info=leastsq(t_fitfunc,pars0)
        if info not in LEASTSQ_SUCCESS or t[1] is None:
            raise Exception(info) 
        chi2=np.sum(t_fitfunc(pars)**2)
        return [np.array(estimator(pars)),[chi2]]   
    try: 
        mean,err,cov=jackknife(dat,tFunc)
    except:
        ret=None
        return sl(sl_key,ret,True)
    
    est_mean,est_err,est_cov=mean[0],err[0],cov[0][0]
    chi2_mean,chi2_err=mean[1][0],err[1][0]
    chi2R_mean,chi2R_err=chi2_mean/Ndof,chi2_err/Ndof
    ret=(est_mean,est_err,est_cov,chi2R_mean,chi2R_err,Ndof,pars)
    return sl(sl_key,ret,True)

def modelAvg(fits):
    '''
    fits=[fit]; fit=(obs_mean,obs_err,chi2R,Ndof)
    '''
    weights=np.exp([-(chi2R*Ndof)/2+Ndof for obs_mean,obs_err,chi2R,Ndof in fits])
    probs=weights/np.sum(weights)
    obs_mean_MA=np.sum(np.array([obs_mean for obs_mean,obs_err,chi2R,Ndof in fits])*probs[:,None],axis=0)
    obs_err_MA=np.sqrt(np.sum(np.array([obs_err**2+obs_mean**2 for obs_mean,obs_err,chi2R,Ndof in fits])*probs[:,None],axis=0)-obs_mean_MA**2)
    return (obs_mean_MA,obs_err_MA,probs)

def renormalize_eVecs(eVecs):
    '''
    eVecs.index=(...,n,i)
    '''
    eVecs_inv=np.linalg.inv(eVecs)
    eVecs_inv=eVecs_inv/np.diagonal(eVecs_inv,axis1=-2,axis2=-1)[...,None,:]
    return np.linalg.inv(eVecs_inv)

def GEVP(Ct,t0List,tList=None,tvList=None):
    '''
    Ct: indexing from t=0
    t0List>=0: t0=t0List
    t0List<0: t0=t-|t0List|
    # Return #
    eVecs (the one to combine source operators): return (time,n,i) but (time,i,n) in the middle
    tv: reference time for getting wave function (Not return wave function if tv is None) 
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

# matplotlib

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
