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

flag_fast=False # If True, certain functions will be speeded up using approximations.

deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)
npRound=lambda dat,n:np.round(np.array(dat).astype(float),n)

def fsolve2(func,x0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res=fsolve(func, x0)[0]
    return res if res!=x0 else np.NaN

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
    return (np.array(y_mean),y_cov)

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
    dat_ens=np.random.multivariate_normal(mean,cov*n,n)
    dat_jk=jackknife(dat_ens)
    # do transformation [pars_jk -> A pars_jk + B] to force pseudo mean and err exactly the same
    mean1,_,cov1=jackmec(dat_jk)
    A=np.sqrt(np.diag(cov)/np.diag(cov1))
    B=mean-A*mean1
    dat_jk=A[None,:]*dat_jk+B[None,:]
    return dat_jk
def jackfit(fitfunc,y_jk,pars0,mask=None,parsExtra_jk=None):
    y_mean,_,y_cov=jackmec(y_jk)
    if mask is not None:
        if mask == 'uncorrelated':
            y_cov=np.diag(np.diag(y_cov))
        else:
            y_cov=y_cov*mask
        
    cho_L_Inv = np.linalg.inv(cholesky(y_cov, lower=True)) # y_cov^{-1}=cho_L_Inv^T@cho_L_Inv
    if parsExtra_jk is None:
        fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(pars)-y_mean)
    else:
        parsExtra_mean=list(np.mean(parsExtra_jk,axis=0))
        fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(list(pars)+parsExtra_mean)-y_mean)
    pars_mean,pars_cov=leastsq(fitfunc_wrapper,pars0,full_output=True)[:2]
    
    if flag_fast == "FastFit": # Generate pseudo jackknife resamples from the single fit rather than doing lots of fits
        n=len(y_jk)
        pars_jk=jackknife_pseudo(pars_mean,pars_cov,n)
    else:
        if parsExtra_jk is None:
            def func(dat):
                fitfunc_wrapper2=lambda pars: cho_L_Inv@(fitfunc(pars)-dat)
                pars=leastsq(fitfunc_wrapper2,pars_mean)[0]
                return pars
            pars_jk=jackmap(func,y_jk)
        else:
            pars_jk=[leastsq(lambda pars: cho_L_Inv@(fitfunc(list(pars)+list(parsExtra))-y),pars_mean)[0] for y,parsExtra in zip(y_jk,parsExtra_jk)]
    
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


# GEVP
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
        
# load

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