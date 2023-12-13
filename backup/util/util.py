import h5py  
import numpy as np
import matplotlib.pyplot as plt
import math,cmath

deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)
npRound=lambda dat,n:np.round(np.array(dat).astype(float),n)

'''
inDat is a list with length Ncfg.
'''
def jackknife(inDat,func=lambda dat:np.mean(np.real(dat),axis=0),delete=1):
    if delete!=1:
        shape=np.array(inDat).shape
        nLeft=(shape[0]//delete)*delete
        shape_new=(delete,shape[0]//delete)+shape[1:]
        dat=np.mean(np.reshape(inDat[:nLeft],shape_new),axis=0)
    else:
        dat=inDat
    tRes=func(dat)
    dim = len(np.array(tRes[0]).shape) + 1
    assert(dim==1 or dim==2)
    
    if dim==1:
        tFunc=func
    elif dim==2:
        lenList=list(map(len,func(dat)))
        tFunc=lambda dat:np.concatenate(func(dat))

    n=len(dat)
    Tn=tFunc(dat)
    Tn1=np.array([tFunc(np.delete(dat,i,axis=0)) for i in range(n)])
    TnBar=np.mean(Tn1,axis=0)
    (tMean,tCov)=(TnBar, np.atleast_2d(np.cov(Tn1.T)*(n-1)*(n-1)/n))
    # (mean,cov)=(n*Tn-(n-1)*TnBar, np.atleast_2d(np.cov(Tn1.T)*(n-1)*(n-1)/n))
    tErr=np.sqrt(np.diag(tCov))
    if dim==1:
        (mean,cov,err)=(tMean,tCov,tErr)
    elif dim==2:
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
        mean=np.array(mean,dtype=object)
        cov=np.array(cov,dtype=object)
        err=np.array(err,dtype=object)
    return (mean,cov,err)
'''
Note: the e-vector is the one to combine source states.
'''
def GEVP(Ct,t0):
    Ct=Ct.astype(np.complex)
    (t_total,N_op,N_op)=Ct.shape
    Ct0=Ct[t0]
    choL=np.linalg.cholesky(Ct0) # Ct0=choL@choL.H
    choLInv=np.linalg.inv(choL)
    w_Ct=choLInv@Ct@np.conj(choLInv).T # w_ denotes tranformed system
    (eVals,w_eVecs)=np.linalg.eig(w_Ct)
    eVals=np.real(eVals)

    # sorting order
    baseSortList=np.arange(N_op)
    tRange=list(range(t0+1,t_total))+list(range(t0,-1,-1))
    # t0+1 case first
    t=t0+1
    sortList=np.argsort(-eVals[t])
    (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])
    for t in tRange[1:]:
        sortList=np.argsort(-eVals[t]) if t>t0 else np.argsort(eVals[t])
        (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])

    # for t in tRange[1:]:
    #     t_sort = t-1 if t>=t0+2 else t+1 if t<=t0-2 else t0+1
    #     inprod=np.abs(np.conj(w_eVecs[t]).T@w_eVecs[t_sort])
    #     sortList=np.argmax(inprod,axis=0)
    #     if len(np.unique(sortList))!=len(sortList):
    #         sortList=baseSortList[np.argmax(inprod,axis=1)]
    #     elif len(np.unique(sortList))!=len(sortList):
    #         inprod=np.abs(np.conj(w_eVecs[t]).T@w_eVecs[t0+1])
    #         sortList=np.argmax(inprod,axis=0)
    #     elif len(np.unique(sortList))!=len(sortList):
    #         sortList=baseSortList[np.argmax(inprod,axis=1)]
    #     elif len(np.unique(sortList))!=len(sortList):
    #         sortList=np.argsort(-eVals[t]) if t>t0 else np.argsort(eVals[t])
    #     (eVals[t],w_eVecs[t])=(eVals[t][sortList],w_eVecs[t][:,sortList])

    eVecs=[]
    for t in range(t_total):
        eVecs.append(np.conj(choLInv).T@w_eVecs[t])
    eVecs=np.array(eVecs)

    return(eVals,eVecs)

# lattice

def getLatList():
    t=['cA2.09.48','cA211.30.32','cA211.53.24']
    print(t)
class LatticeEnsemble:
    hbarc = 1/197.3 # hbarc = fm * MeV
    def __init__(self, ensemble):
        if ensemble == 'cA2.09.48':
            self.a=0.0938; self.L=4.50; self.ampi=0.06208; self.amN=0.4436
            self.info='cSW=1.57551, beta=2.1, Nf=2, V=48^3*96'
        elif ensemble == 'cA211.30.32':
            self.a=0.0947; self.L=3.03; self.ampi=0.12530; self.amN=0.5073
        elif ensemble == 'cA211.53.24':
            self.a=0.0947; self.L=2.27; self.ampi=0.16626; self.amN=0.56
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

LargeFigFlag=False

def getFigAxs(Nrow,Ncol,Lrow=4,Lcol=5):
    global LargeFigFlag
    if LargeFigFlag:
        Lrow*=2; Lcol*=2
    fig, axs = plt.subplots(Nrow, Ncol, figsize=(Lcol*Ncol, Lrow*Nrow), squeeze=False)
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