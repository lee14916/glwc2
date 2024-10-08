import h5py  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math,cmath
import pickle
from scipy.optimize import leastsq, curve_fit
from scipy.linalg import solve_triangular,cholesky
from inspect import signature

flag_fast=False

deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)
npRound=lambda dat,n:np.round(np.array(dat).astype(float),n)


prefactorDeep=lambda dat,prefactor:np.real(prefactor*dat) if type(dat)==np.ndarray else [prefactorDeep(ele,prefactor) for ele in dat]
meanDeep=lambda dat:np.mean(dat,axis=0) if type(dat)==np.ndarray else [meanDeep(ele) for ele in dat]
def jackknife(in_dat,in_func=lambda dat:np.mean(np.real(dat),axis=0),d:int=1,outputFlatten=False):
    '''
    - in_dat: any-dimensional list of ndarrays. Each ndarray in the list has 0-axis for cfgs
    - in_func: dat -> estimator
    - Estimator: number or 1d-list/array or 2d-list/array or 1d-list of 1d-arrays
    - d: jackknife delete parameter
    ### return: mean,err,cov
    - mean,err: estimator's format reformatted to 1d-list of 1d-arrays
    - cov: 2d-list of 2d-arrays
    '''
    if flag_fast:
        getNcfg=lambda dat: len(dat) if type(dat)==np.ndarray else getNcfg(dat[0])
        n=getNcfg(in_dat)
        d=n//300
    
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
        
    # getNcfg
    getNcfg=lambda dat: len(dat) if type(dat)==np.ndarray else getNcfg(dat[0])
        
    # delete i 
    delete= lambda dat,i: np.delete(dat,i,axis=0) if type(dat)==np.ndarray else [delete(ele,i) for ele in dat]
    
    # jackknife     
    n=getNcfg(dat)
    Tn1=np.array([func(delete(dat,i)) for i in range(n)])
    TnBar=np.mean(Tn1,axis=0)
    (tMean,tCov)=(TnBar, np.atleast_2d(np.cov(Tn1.T)*(n-1)*(n-1)/n))
    # Tn=func(dat); (mean,cov)=(n*Tn-(n-1)*TnBar, np.atleast_2d(np.cov(Tn1.T)*(n-1)*(n-1)/n)) # bias improvement (not suitable for fit)
    tErr=np.sqrt(np.diag(tCov))
    
    if outputFlatten:
        return (tMean,tErr,tCov)
    
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
    return (mean,err,cov)

flag_fit_cov2err=False
def fit(dat,func,fitfunc,estimator=None,pars0=None,mask_cov=None,jk=True):
    '''
    dat: raw data
    func: dat -> y_exp
    fitfunc: par -> y_model
    return: (mean,err,cov,chi2R,chi2R_err,Ndof)
    '''
    Npar=len(signature(fitfunc).parameters)
    pars0=pars0 if pars0 is not None else [1 for i in range(Npar)]

    _,_,cov0=jackknife(dat,func,outputFlatten=True)
    if flag_fit_cov2err:
        cov0=np.diag(np.diag(cov0))
    if mask_cov is not None:
        cov0=cov0*mask_cov
    cho_L_Inv = np.linalg.inv(cholesky(cov0, lower=True))
    Ny=len(cov0)
    Ndof=Ny-Npar
    
    def tFunc(dat):
        y_exp=np.hstack(func(dat))
        t_fitfunc=lambda pars: cho_L_Inv@(np.hstack(fitfunc(*pars))-y_exp)
        pars,pars_cov=leastsq(t_fitfunc,pars0,full_output=True)[:2]
        chi2=np.sum(t_fitfunc(pars)**2)
        return (pars,pars_cov,chi2)
    mean,cov,chi2=tFunc(dat)
    if estimator is None and (flag_fast or not jk):
        if cov is None:
            cov=np.zeros((len(mean),len(mean)))+np.inf
        res=(mean,np.sqrt(np.diag(cov)),cov,chi2/Ndof,None,Ndof)
        return res
    pars0=mean

    def tFunc(dat):
        y_exp=np.hstack(func(dat))
        t_fitfunc=lambda pars: cho_L_Inv@(np.hstack(fitfunc(*pars))-y_exp)
        pars,_=leastsq(t_fitfunc,pars0)
        chi2=np.sum(t_fitfunc(pars)**2)
        return [pars if estimator is None else estimator(pars),[chi2]]    
    mean,err,cov=jackknife(dat,tFunc)
    
    pars_mean,pars_err,pars_cov=mean[0],err[0],cov[0][0]
    chi2_mean,chi2_err=mean[1][0],err[1][0]
    chi2R_mean,chi2R_err=chi2_mean/Ndof,chi2_err/Ndof
    res=(pars_mean,pars_err,pars_cov,chi2R_mean,chi2R_err,Ndof)
    
    return res

def modelAvg(fits):
    '''
    fits=[fit]; fit=(obs_mean,obs_err,chi2R,Ndof)
    '''
    weights=np.exp([-(chi2R*Ndof)/2+Ndof for obs_mean,obs_err,chi2R,Ndof in fits])
    probs=weights/np.sum(weights)
    obs_mean_MA=np.sum(np.array([obs_mean for obs_mean,obs_err,chi2R,Ndof in fits])*probs[:,None],axis=0)
    obs_err_MA=np.sqrt(np.sum(np.array([obs_err**2+obs_mean**2 for obs_mean,obs_err,chi2R,Ndof in fits])*probs[:,None],axis=0)-obs_mean_MA**2)
    return (obs_mean_MA,obs_err_MA,probs)

def GEVP(Ct,t0,compQ=False):
    '''
    t0>=0: tRef=t0
    t0<0: tRef=t-|t0|
    eVecs: (time,i,n)
    Note: the e-vector is the one to combine source states.
    '''
    Ct=Ct.astype(complex)
    (t_total,N_op,N_op)=Ct.shape
    Ct0=np.roll(Ct,-t0,axis=0) if t0<0 else np.array([Ct[t0] for t in range(t_total)])
    choL=np.linalg.cholesky(Ct0) # Ct0=choL@choL.H
    choLInv=np.linalg.inv(choL)
    choLInvDag=np.conj(np.transpose(choLInv,[0,2,1]))
    w_Ct=choLInv@Ct@choLInvDag
    (eVals,w_eVecs)=np.linalg.eig(w_Ct)
    eVals=np.real(eVals)

    # sorting order
    if t0<0:
        for t in range(t_total):
            sortList=np.argsort(-eVals[t])
            (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])
    else:
        baseSortList=np.arange(N_op)
        tRange=list(range(t0+1,t_total))+list(range(t0,-1,-1))
        # t0+1 case first
        t=t0+1
        sortList=np.argsort(-eVals[t])
        (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])
        for t in tRange[1:]:
            sortList=np.argsort(-eVals[t]) if t>t0 else np.argsort(eVals[t])
            (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])

    eVecs=choLInvDag@w_eVecs
    
    if compQ:
        powers=np.array([-(t+t0)/t0 for t in range(t_total)] if t0<0 else [t0/(t-t0) if t!=t0 else 9999 for t in range(t_total)])
        t=np.conj(np.transpose(eVecs,[0,2,1]))@Ct0@eVecs
        t=np.real(t[:,range(N_op),range(N_op)])
        fn=np.sqrt( t / (eVals**powers[:,None]))
        eVecs_normalized=eVecs/fn[:,None,:]
        A=np.transpose(eVecs_normalized,[0,2,1]) # A_ni
        AInv=np.linalg.inv(A) # (A^-1)_in
        compQ=AInv
        return (eVals,eVecs,compQ)

    return (eVals,eVecs)

# def GEVP(Ct,t0):
#     '''
#     Note: the e-vector is the one to combine source states.
#     '''
#     Ct=Ct.astype(complex)
#     (t_total,N_op,N_op)=Ct.shape
#     Ct0=Ct[t0]
#     choL=np.linalg.cholesky(Ct0) # Ct0=choL@choL.H
#     choLInv=np.linalg.inv(choL)
#     w_Ct=choLInv@Ct@np.conj(choLInv).T # w_ denotes tranformed system
#     (eVals,w_eVecs)=np.linalg.eig(w_Ct)
#     eVals=np.real(eVals)

#     # sorting order
#     baseSortList=np.arange(N_op)
#     tRange=list(range(t0+1,t_total))+list(range(t0,-1,-1))
#     # t0+1 case first
#     t=t0+1
#     sortList=np.argsort(-eVals[t])
#     (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])
#     for t in tRange[1:]:
#         sortList=np.argsort(-eVals[t]) if t>t0 else np.argsort(eVals[t])
#         (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])

#     # for t in tRange[1:]:
#     #     t_sort = t-1 if t>=t0+2 else t+1 if t<=t0-2 else t0+1
#     #     inprod=np.abs(np.conj(w_eVecs[t]).T@w_eVecs[t_sort])
#     #     sortList=np.argmax(inprod,axis=0)
#     #     if len(np.unique(sortList))!=len(sortList):
#     #         sortList=baseSortList[np.argmax(inprod,axis=1)]
#     #     elif len(np.unique(sortList))!=len(sortList):
#     #         inprod=np.abs(np.conj(w_eVecs[t]).T@w_eVecs[t0+1])
#     #         sortList=np.argmax(inprod,axis=0)
#     #     elif len(np.unique(sortList))!=len(sortList):
#     #         sortList=baseSortList[np.argmax(inprod,axis=1)]
#     #     elif len(np.unique(sortList))!=len(sortList):
#     #         sortList=np.argsort(-eVals[t]) if t>t0 else np.argsort(eVals[t])
#     #     (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])

#     eVecs=[]
#     for t in range(t_total):
#         eVecs.append(np.conj(choLInv).T@w_eVecs[t])
#     eVecs=np.array(eVecs)

#     return(eVals,eVecs)

# lattice

def getLatList():
    t=['cA2.09.48','cA211.530.24']
    print(t)
class LatticeEnsemble:
    hbarc = 1/197.3 # hbarc = fm * MeV
    def __init__(self, ensemble):
        if ensemble == 'cA2.09.48':
            self.a=0.0938; self.L=4.50; self.ampi=0.06208; self.amN=0.4436
            self.info='cSW=1.57551, beta=2.1, Nf=2, V=48^3*96'
            self.amu=0.0009
            self.ZA=0.791; self.ZP=0.500; self.ZS=0.661
        # elif ensemble == 'cA211.30.32':
        #     self.a=0.0947; self.L=3.03; self.ampi=0.12530; self.amN=0.5073
        elif ensemble == 'cA211.530.24':
            self.a=0.0947; self.L=2.27; self.ampi=0.16626; self.amN=0.56
            self.amu=0.0053
            self.ZA=0.7501; self.ZP=0.4654; self.ZS=0.6715; self.ZV=0.7061; self.ZT=0.8030 # all for non-singlet
        else:
            print(ensemble+' not implemented')
        self.aInv=1/(self.a*self.hbarc); self.tpiL=(2*math.pi)/(self.L*self.hbarc); 
        self.mpiL=self.ampi * self.aInv * self.L * self.hbarc
        self.mpi=self.ampi*self.aInv; self.mN=self.amN*self.aInv
    def getaEpi(self, n2):
        return np.sqrt(self.mpi**2+self.tpiL**2*n2)*self.a*self.hbarc
    def getaEN(self, n2):
        return np.sqrt(self.mN**2+self.tpiL**2*n2)*self.a*self.hbarc
    def amass2mass(self, amass):
        return amass*self.aInv
    
# matplotlib

def getFigAxs(Nrow,Ncol,Lrow=None,Lcol=None,scale=1,**kwargs):
    if (Lrow,Lcol)==(None,None):
        Lcol,Lrow=mpl.rcParams['figure.figsize']
        Lrow*=scale; Lrow*=scale
        # if (Nrow,Ncol)==(1,1):
        #     Lcol*=1.5; Lrow*=1.5
    fig, axs = plt.subplots(Nrow, Ncol, figsize=(Lcol*Ncol, Lrow*Nrow), squeeze=False,**kwargs)
    return fig, axs
    
def addRowHeader(axs,rows):
    pad=5
    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
        
def addColHeader(axs,cols):
    pad=5
    for ax, col in zip(axs[0,:], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

        
#######################################################################################

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
        [[],['pi0i']]],
    [['B3pt','W3pt','Z3pt'],{'N','j','pib'},\
        [[]]],
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


def load(path):
    print('loading: '+path)
    data_load={}
    with h5py.File(path) as f:
        datasets=[]
        def visit_function(name,node):
            if isinstance(node, h5py.Dataset):
                datasets.append(name)
                # print(len(datasets),name,end='\r')
        f.visititems(visit_function)
              
        N=len(datasets)
        for i,dataset in enumerate(datasets):
            data_load[dataset]=f[dataset][()]
            print(str(i+1)+'/'+str(N)+': '+dataset,end='                           \r')
        print()

    def op_new(op,fla):
        t=op.split(';')
        t[-1]=fla
        return ';'.join(t)
    gjList=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt']
    diags=set([dataset.split('/')[1] for dataset in list(data_load.keys()) if 'diags' in dataset])
    opabsDic={}
    for diag in diags:
        opabsDic[diag]=[opab.decode() for opab in data_load['/'.join(['diags',diag,'opabs'])]]
        
    data={'2pt':{},'3pt':{},'VEV':{}}
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
            data[npt][diag]=data_load[dataset]
          
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
def op2opl(op,l):
    t=op.split(';')
    t[-2]=l
    return ';'.join([t])
def needsVEV(opa,opb,insert):
    flag=True
    _,pta,irrepa,_,la,_=opa.split(';'); _,ptb,irrepb,_,lb,_=opb.split(';')
    if pta!=ptb or irrepa!=irrepb or la!=lb:
        flag=False
    return flag

path='aux/group_coeffs.pkl'
with open(path,'rb') as f:
    coeff=pickle.load(f)
    
def getCoeMat(pt):
    t=np.zeros([2,2],dtype=complex)
    for op,val in coeff[getop(pt,'l1',('a','N'))]:
        t[0,int(op.split(';')[2])]=val
    for op,val in coeff[getop(pt,'l2',('a','N'))]:
        t[1,int(op.split(';')[2])]=val
    return np.linalg.inv(t)

gtCj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':1,
      'sgmxy':-1} # gt G^dag gt = (gtCj) G