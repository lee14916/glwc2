'''
nohup python3 -u analysis3_SVD.py -e b > log/analysis3_SVD_b.out &
'''
import os,sys,warnings,click
import h5py, pandas
import numpy as np
np.seterr(invalid=['ignore','warn'][0])
np.set_printoptions(legacy='1.25')
import math,cmath,pickle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit,fsolve
import matplotlib as mpl
import matplotlib.pyplot as plt
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
import util as yu
yu.flag_fast=False

enss=['b','c','d']
ens2full={'a24':'cA211.53.24','a':'cA2.09.48','b':'cB211.072.64','c':'cC211.060.80','d':'cD211.054.96','e':'cE211.044.112'}
ens2label={'a24':'A24','a':'A48','b':'B64','c':'C80','d':'D96','e':'E112'}
ens2a={'a24':0.0908,'a':0.0938,'b':0.07957,'c':0.06821,'d':0.05692,'e':0.04892} # fm
ens2N={'a24':24,'a':48,'b':64,'c':80,'d':96,'e':112}
ens2N_T={'a24':24*2,'a':48*2,'b':64*2,'c':80*2,'d':96*2,'e':112*2}

hbarc = 1/197.3
ens2aInv={ens:1/(ens2a[ens]*hbarc) for ens in enss} # MeV

def find_t_cloest(ens,t):
    return round(t/ens2a[ens])

baseFigPath=f'fig/analysis3/'

stouts_global=[7,10,13,20]
stouts=stouts_global

ens2Njk={}
for ens in enss:
    path=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run3/{ens2full[ens]}/data_merge/data.h5'
    with h5py.File(path) as f:
        t=f['discNJN_j+;g{m,Dn};tl.h5']['notes'][:]
        projs=t[-1].decode().split('=')[-1][1:-1].split(',')
        t=f['j.h5/inserts'][:]
        inserts=[ele.decode() for ele in t]
        ens2Njk[ens]=len(f['N.h5/data/N_N'])
        
        inds_equal=[i for i,insert in enumerate(inserts) if insert[0]==insert[1]]
        inds_unequal=[i for i,insert in enumerate(inserts) if insert[0]!=insert[1]]
        
print(projs)
print(inserts)


###
data={}
for ens in enss:
    path=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run3/{ens2full[ens]}/data_merge/data.h5'
    with h5py.File(path) as f:
        moms=f['N.h5/moms'][:]
        moms=[tuple(mom) for mom in moms]
        ind=moms.index((0,0,0))
        
        data[ens]=yu.jackknife(np.real(f['N.h5/data/N_N'][:,:,ind]))

propThreshold=0.1
# propThreshold=None

chi2Size=9
settings={}

func_C2pt_1st=lambda t,E0,c0: c0*np.exp(-E0*t)
func_C2pt_2st=lambda t,E0,c0,dE1,rc1: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t))
func_C2pt_3st=lambda t,E0,c0,dE1,rc1,dE2,rc2: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t) + rc2*np.exp(-dE2*t))
func_mEff_1st=lambda t,E0: np.log(func_C2pt_1st(t,E0,1)/func_C2pt_1st(t+1,E0,1))
func_mEff_2st=lambda t,E0,dE1,rc1: np.log(func_C2pt_2st(t,E0,1,dE1,rc1)/func_C2pt_2st(t+1,E0,1,dE1,rc1))
func_mEff_3st=lambda t,E0,dE1,rc1,dE2,rc2: np.log(func_C2pt_3st(t,E0,1,dE1,rc1,dE2,rc2)/func_C2pt_3st(t+1,E0,1,dE1,rc1,dE2,rc2))

def run(ens,figname=None):
    corrQ=True; meffQ=True
    xunit=ens2a[ens]; yunit=ens2aInv[ens]/1000
    fig, axd = plt.subplot_mosaic([['f1','f1','f1'],['f2','f2','f3']],figsize=(24,10))
    (ax1,ax2,ax3)=(axd[key] for key in ['f1','f2','f3'])
    # if meffQ:
    #     fig.suptitle('Correlated fit to meff' if corrQ else 'Uncorrelated fit to meff',fontsize=44)
    # else:
    #     fig.suptitle('Correlated fit to C2pt' if corrQ else 'Uncorrelated fit to C2pt',fontsize=44)
    
    ax1.set_xlabel(r'$t$ [fm]')
    ax2.set_xlabel(r'$t_{\mathrm{min}}$ [fm]')
    ax3.set_xlabel(r'$t_{\mathrm{min}}$ [fm]')
    ax1.set_ylabel(r'$m_N^{\mathrm{eff}}$ [GeV]')
    ax2.set_ylabel(r'$m_N$ [GeV]')
    ax3.set_ylabel(r'$E_1$ [GeV]')
    ax1.set_ylim(settings['ylim1'])
    ax2.set_ylim(settings['ylim2'])
    ax3.set_ylim(settings['ylim3'])
    ax1.set_xlim(settings['xlim1'])
    ax2.set_xlim(settings['xlim2'])
    ax3.set_xlim(settings['xlim3'])
    
    mN_exp=0.938; mp_exp,mn_exp=(0.93827,0.93957)
    ax1.axhline(y=mN_exp,color='black',linestyle = '--', marker='')
    ax2.axhline(y=mN_exp,color='black',linestyle = '--', marker='', label=r'$m_N^{\mathrm{exp}}=$'+'%0.3f'%mN_exp)
    C2pt_jk=data[ens]
    C2pt_mean,C2pt_err=yu.jackme(C2pt_jk)
    C2pt_rela=np.abs(C2pt_err/C2pt_mean)
    func=lambda C2pt: np.log(C2pt/np.roll(C2pt,-1,axis=0))
    mEff_jk=yu.jackmap(func,C2pt_jk)
    (mEff_mean,mEff_err)=yu.jackme(mEff_jk)
    mEff_rela=np.abs(mEff_err/mEff_mean)
    temp=[(i,rela) for i,rela in enumerate(mEff_rela if meffQ else C2pt_rela) if rela>0.2 and i!=0]
    fitmax=temp[0][0]-1 if len(temp)!=0 else len(C2pt_mean)-1
    
    tmin=1; tmax=fitmax+1
    plt_x=np.arange(tmin,tmax)*xunit; plt_y=mEff_mean[tmin:tmax]*yunit; plt_err=mEff_err[tmin:tmax]*yunit
    ax1.errorbar(plt_x,plt_y,plt_err,color='black',fmt='s')

    pars0_initial=[0.4,0.5,2,0.8,1] if meffQ else [0.4,1e-8,0.5,2,0.8,1]
    DNpar=1 if meffQ else 0
    
    fits_all=[]
    # 1st fits
    color='r'
    fitmins=settings['fitmins_1st']
    pars0=pars0_initial[:2-DNpar]
    fits=[]
    for fitmin in fitmins:
        tList=np.arange(fitmin,fitmax,2)
        def fitfunc(pars):
            if meffQ:
                return func_mEff_1st(tList,*pars)
            return func_C2pt_1st(tList,*pars)
        y_jk=mEff_jk[:,tList] if meffQ else C2pt_jk[:,tList]
        pars_jk,chi2_jk,Ndof=yu.jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
        pars0=np.mean(pars_jk,axis=0)
        fits.append([fitmin,pars_jk,chi2_jk,Ndof])
        fits_all.append([('1st',fitmin),pars_jk[:,:1],chi2_jk,Ndof])
        
    pars_jk,props_jk=yu.jackMA(fits)
    props_mean=np.mean(props_jk,axis=0)
    ind_mpf=np.argmax(np.mean(props_jk,axis=0))    
    pars_mean,pars_err=yu.jackme(pars_jk)
    pars0=pars_mean
    plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_err=pars_err[0]*yunit
    ax2.fill_between(plt_x,plt_y-plt_err,plt_y+plt_err,color=color,alpha=0.2,label=r'$m_N^{\mathrm{1st}}=$'+yu.un2str(plt_y,plt_err))
    for i,fit in enumerate(fits):
        fitmin,pars_jk,chi2_jk,Ndof=fit; prop=props_mean[i]
        (pars_mean,pars_err)=yu.jackme(pars_jk)
        chi2R=np.mean(chi2_jk)/Ndof
        showQ = i==ind_mpf if propThreshold is None else prop>propThreshold
        
        plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_err=pars_err[0]*yunit
        ax2.errorbar(plt_x,plt_y,plt_err,fmt='s',color=color,mfc='white' if showQ else None)
        ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
        ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_err-chi2_shift),color=color,size=chi2Size,ha='center')
        if propThreshold is not None and prop>propThreshold:
            ax2.annotate(f"{int(prop*100)}%",(plt_x,plt_y+plt_err+chi2_shift*0.5),color=color,size=chi2Size,ha='center')
            
    # 2st fits
    color='g'
    fitmins=settings['fitmins_2st']
    pars0=np.hstack([pars0,pars0_initial[2-DNpar:4-DNpar]])
    fits=[]
    for fitmin in fitmins:
        # print(2,fitmin)
        tList=np.arange(fitmin,fitmax,2)
        def fitfunc(pars):
            if meffQ:
                return func_mEff_2st(tList,*pars)
            return func_C2pt_2st(tList,*pars)
        y_jk=mEff_jk[:,tList] if meffQ else C2pt_jk[:,tList]
        pars_jk,chi2_jk,Ndof=yu.jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
        pars0=np.mean(pars_jk,axis=0)
        fits.append([fitmin,pars_jk,chi2_jk,Ndof])
        fits_all.append([('2st',fitmin),pars_jk[:,:1],chi2_jk,Ndof])
    pars_jk,props_jk=yu.jackMA(fits)
    props_mean=np.mean(props_jk,axis=0)
    res=pars_jk.copy()
    ind_mpf=np.argmax(np.mean(props_jk,axis=0))    
    pars0=yu.jackme(pars_jk)[0]
    pars_jk[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
    pars_mean,pars_err=yu.jackme(pars_jk)
    plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_err=pars_err[0]*yunit
    ax2.fill_between(plt_x,plt_y-plt_err,plt_y+plt_err,color=color,alpha=0.2, label=r'$m_N^{\mathrm{2st}}=$'+yu.un2str(plt_y,plt_err))
    plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_err=pars_err[1]*yunit
    ax3.fill_between(plt_x,plt_y-plt_err,plt_y+plt_err,color=color,alpha=0.2, label=r'$E_1^{\mathrm{2st}}=$'+yu.un2str(plt_y,plt_err))
    for i,fit in enumerate(fits):
        fitmin,pars_jk,chi2_jk,Ndof=fit; prop=props_mean[i]
        pars_jk[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
        (pars_mean,pars_err)=yu.jackme(pars_jk)
        chi2R=np.mean(chi2_jk)/Ndof
        showQ = i==ind_mpf if propThreshold is None else prop>propThreshold
        
        plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_err=pars_err[0]*yunit
        ax2.errorbar(plt_x,plt_y,plt_err,fmt='s',color=color,mfc='white' if showQ else None)
        ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
        ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_err-chi2_shift),color=color,size=chi2Size,ha='center')
        if propThreshold is not None and prop>propThreshold:
            ax2.annotate(f"{int(prop*100)}%",(plt_x,plt_y+plt_err+chi2_shift*0.5),color=color,size=chi2Size,ha='center')
        
        plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_err=pars_err[1]*yunit
        ax3.errorbar(plt_x,plt_y,plt_err,fmt='s',color=color,mfc='white' if showQ else None)
        ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
        ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_err-chi2_shift),color=color,size=chi2Size,ha='center')
        if propThreshold is not None and prop>propThreshold:
            ax3.annotate(f"{int(prop*100)}%",(plt_x,plt_y+plt_err+chi2_shift*0.5),color=color,size=chi2Size,ha='center')

    # 3st fits
    color='b'
    fitmins=settings['fitmins_3st']
    pars0=np.hstack([pars0,pars0_initial[4-DNpar:6-DNpar]])
    if ens=='c' and (corrQ,meffQ)==(False,False):
        pars0=[3.25069715e-01, 1.88384811e-09, 1.78883939e-01, 6.35351339e-01, 6.98775484e-01, 4.58702896e+01]
    # elif ens=='d' and (corrQ,meffQ)==(False,False):
    #     pars=[2.72824764e-01, 3.72721072e-10, 1.84246641e-01, 7.65383428e-01, 6.98775484e-01, 4.58702896e+01]
    fits=[]
    for fitmin in fitmins:
        # print(3,fitmin)
        tList=np.arange(fitmin,fitmax,2)
        def fitfunc(pars):
            if meffQ:
                return func_mEff_3st(tList,*pars)
            return func_C2pt_3st(tList,*pars)
        y_jk=mEff_jk[:,tList] if meffQ else C2pt_jk[:,tList]
        pars_jk,chi2_jk,Ndof=yu.jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
        pars0=np.mean(pars_jk,axis=0)
        fits.append([fitmin,pars_jk,chi2_jk,Ndof])
        fits_all.append([('3st',fitmin),pars_jk[:,:1],chi2_jk,Ndof])
    pars_jk,props_jk=yu.jackMA(fits)
    props_mean=np.mean(props_jk,axis=0)
    ind_mpf=np.argmax(np.mean(props_jk,axis=0))    
    pars0=yu.jackme(pars_jk)[0]
    # print(pars0)
    pars_jk[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
    pars_mean,pars_err=yu.jackme(pars_jk)
    plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_err=pars_err[0]*yunit
    ax2.fill_between(plt_x,plt_y-plt_err,plt_y+plt_err,color=color,alpha=0.2, label=r'$m_N^{\mathrm{3st}}=$'+yu.un2str(plt_y,plt_err))
    plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_err=pars_err[1]*yunit
    ax3.fill_between(plt_x,plt_y-plt_err,plt_y+plt_err,color=color,alpha=0.2, label=r'$E_1^{\mathrm{3st}}=$'+yu.un2str(plt_y,plt_err))    
    for i,fit in enumerate(fits):
        fitmin,pars_jk,chi2_jk,Ndof=fit; prop=props_mean[i]
        pars_jk[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
        (pars_mean,pars_err)=yu.jackme(pars_jk)
        chi2R=np.mean(chi2_jk)/Ndof
        showQ = i==ind_mpf if propThreshold is None else prop>propThreshold
        
        plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_err=pars_err[0]*yunit
        ax2.errorbar(plt_x,plt_y,plt_err,fmt='s',color=color,mfc='white' if showQ else None)
        ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
        ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_err-chi2_shift),color=color,size=chi2Size,ha='center')
        if propThreshold is not None and prop>propThreshold:
            ax2.annotate(f"{int(prop*100)}%",(plt_x,plt_y+plt_err+chi2_shift*0.5),color=color,size=chi2Size,ha='center')
        
        plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_err=pars_err[1]*yunit
        ax3.errorbar(plt_x,plt_y,plt_err,fmt='s',color=color,mfc='white' if showQ else None)
        ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
        ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_err-chi2_shift),color=color,size=chi2Size,ha='center') 
        if propThreshold is not None and prop>propThreshold:
            ax3.annotate(f"{int(prop*100)}%",(plt_x,plt_y+plt_err+chi2_shift*0.5),color=color,size=chi2Size,ha='center')
        
    color='orange'
    pars_jk,props_jk=yu.jackMA(fits_all)
    ind_mpf=np.argmax(np.mean(props_jk,axis=0))
    pars_mean,pars_err=yu.jackme(pars_jk)
    plt_x=settings['xlim2']; plt_y=pars_mean[0]*yunit; plt_err=pars_err[0]*yunit
    ax2.fill_between(plt_x,plt_y-plt_err,plt_y+plt_err,color=color,alpha=0.2, label=r'$m_N^{\mathrm{nst}}=$'+yu.un2str(plt_y,plt_err) + f'; MPF: {fits_all[ind_mpf][0][0]}')    
    
    ax2.legend(loc=(0.6,0.5),fontsize=12)
    ax3.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
    return res

res_c2ptN0={}
for ens in enss[:]:
    if ens=='b':
        settings={
            'fitmins_1st':range(8,24+1),
            'fitmins_2st':range(1,10+1),
            'fitmins_3st':range(1,4+1),
            'ylim1':[0.6,1.7],
            'ylim1':[0.88,1.08],
            'ylim2':[0.85,1.1],
            'ylim3':[0.85,3.0],
            'xlim1':[0,2.7],
            'xlim2':[0,2.2],
            'xlim3':[0,1.1],
        }
    elif ens=='c':
        settings={
            'fitmins_1st':range(8,29+1),
            'fitmins_2st':range(1,17+1),
            'fitmins_3st':range(1,7+1),
            'ylim1':[0.6,1.7],
            'ylim1':[0.88,1.08],
            'ylim2':[0.85,1.1],
            'ylim3':[0.85,3.0],
            'xlim1':[0,2.7],
            'xlim2':[0,2.2],
            'xlim3':[0,1.1],
        }
    elif ens=='d':
        settings={
            'fitmins_1st':range(8,34+1),
            'fitmins_2st':range(1,20+1),
            'fitmins_3st':range(1,6+1),
            'ylim1':[0.6,1.7],
            'ylim1':[0.88,1.08],
            'ylim2':[0.85,1.1],
            'ylim3':[0.85,3.0],
            'xlim1':[0,2.7],
            'xlim2':[0,2.2],
            'xlim3':[0,1.1],
        }
    res_c2ptN0[ens]=run(ens,figname=f'{baseFigPath}fig_ignore/c2ptN0_{ens}.pdf')
    
ens2mN={}
for ens in enss:
    ens2mN[ens]=res_c2ptN0[ens][:,0]
###
import sympy as sp
from sympy import sqrt
from itertools import permutations

id=np.eye(4)
g1=np.array([[0, 0, 0, 1j],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [-1j, 0, 0, 0]])

g2=np.array([[0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0]])

g3=np.array([[0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0]])

g4=np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]])

g5 = g1@g2@g3@g4
gm = np.array([g1, g2, g3, g4])
sgm = np.array([[(gm[mu]@gm[nu] - gm[nu]@gm[mu])/2 for nu in range(4)] for mu in range(4)])

G0 = (id + g4) / 4
G1 = 1j * g5 @ g1 @ G0
G2 = 1j * g5 @ g2 @ G0
G3 = 1j * g5 @ g3 @ G0
G = [G1, G2, G3, G0]

insert2ind={'x':0,'y':1,'z':2,'t':3}
def ME2FF(m,pvec,pvec1,proj,insert):
    Gn={'P0':G0,'Px':G1,'Py':G2,'Pz':G3}[proj]
    mu,nu=insert
    mu=insert2ind[mu]; nu=insert2ind[nu]
    
    px,py,pz=pvec
    p1x,p1y,p1z=pvec1
    
    if m==sp.symbols('m'):
        pt=sp.symbols('pt')
        p1t=sp.symbols('p1t')
    else:
        pt=1j*np.sqrt(px**2+py**2+pz**2+m**2)
        p1t=1j*np.sqrt(p1x**2+p1y**2+p1z**2+m**2)
        
    p=np.array([px,py,pz,pt])
    p1=np.array([p1x,p1y,p1z,p1t])

    pS=np.sum(gm*p[:,None,None],axis=0)
    p1S=np.sum(gm*p1[:,None,None],axis=0)
    Px, Py, Pz, Pt = p + p1
    qx, qy, qz, qt = p - p1
    P=np.array([Px,Py,Pz,Pt])
    q=np.array([qx,qy,qz,qt])
    Q2 = -2*m**2 - 2*p1.dot(p)
    
    #==============================
    factorA= 1j; factorB= -1j; factorC=1
    factorBase=1/(4*m**2); factorSgm=1
    
    la=(gm[mu]*P[nu]/2+gm[nu]*P[mu]/2)/2-(np.sum(gm*P[:,None,None]/2,axis=0))*id[mu,nu]/4
    lb=(1j/(2*m))*((np.einsum('rab,r->ab',sgm[mu],q)*P[nu]/2+np.einsum('rab,r->ab',sgm[nu],q)*P[mu]/2)/2-np.einsum('srab,r,s->ab',sgm,q,P/2)*id[mu,nu]/4)*factorSgm
    lc=(id/m)*(q[mu]*q[nu]-Q2/4*id[mu,nu])
    
    res=np.array([factorBase*factor*np.trace(Gn@(-1j*p1S+m*id)@Lambda@(-1j*pS+m*id)) for Lambda,factor in zip([la,lb,lc],[factorA,factorB,factorC])])
    
    if m==sp.symbols('m'):
        xE = sp.symbols('E')
        for t in res:
            t=t.subs({p1x:0,p1y:0,p1z:0,p1t:1j*m,pt:1j*xE})
            # t=t.subs({px:sqrt(xE**2-m**2-py**2-pz**2)})
            t=sp.expand(sp.sympify(t))
            print(t)
        print()
        return
    
    return res

def nonzeroQ(mom,proj,insert):
    n1vec=np.array(mom[:3]); nqvec=np.array(mom[3:6])
    nvec=n1vec+nqvec
    
    m=938/ens2aInv[ens]; L=ens2N[ens]
    pvec=nvec*(2*np.pi/L); p1vec=n1vec*(2*np.pi/L)
    
    res=ME2FF(m,pvec,p1vec,proj,insert)
    tr=np.sum(np.abs(np.real(res))); ti=np.sum(np.abs(np.imag(res)))
    threshold=1e-8
    return (tr>threshold,ti>threshold)

def rotateMPI(rot,mom,proj,insert):
    sx,sy,sz,xyz=rot; signs=[sx,sy,sz,1]
    ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
    xyzt=['x','y','z','t']
    xyzt2={'x':xyzt[ix],'y':xyzt[iy],'z':xyzt[iz],'t':'t'}
    
    mom1=[sx*mom[iix],sy*mom[iiy],sz*mom[iiz],sx*mom[iix+3],sy*mom[iiy+3],sz*mom[iiz+3]]
    proj1='P0' if proj=='P0' else f'P{xyzt2[proj[1]]}'
    insert1=f'{xyzt2[insert[0]]}{xyzt2[insert[1]]}'
    insert1=insert1 if insert1 in inserts else insert1[1]+insert1[0]
    return [mom1,proj1,insert1]

def sortFunc(mpi):
    return ''.join(mpi)

def useQ(mom,proj,insert):
    r,i=nonzeroQ(mom,proj,insert)
    if (r,i)==(False,False):
        return (False,False)
    if insert == 'tt': # traceless makes tt=-xx-yy-zz
        return (False,False)
    
    elements=[(sx,sy,sz,xyz) for sx in [1,-1] for sy in [1,-1] for sz in [1,-1] for xyz in permutations([0, 1, 2], 3)]
    mpis=[rotateMPI(e,mom,proj,insert) for e in elements]
    pis=[('_'.join([str(e) for e in m]),p,i) for m,p,i in mpis if np.all(list(m[3:])==mom[3:])]
    pis=list(set(pis))
    pis.sort(key=sortFunc)
    if ('_'.join([str(e) for e in mom]),proj,insert) != pis[-1]:
        return (False,False)
    return (r,i)

ME2FF(sp.symbols('m'),sp.symbols('px py pz'),sp.symbols('p1x p1y p1z'),'Px','xy')

ME2FF(1,[0,0,0],[0,0,0],'P0','tt')

nonzeroQ([0,0,0,0,0,0],'P0','tt')

useQ([0,0,0,0,0,0],'P0','zz')

useQ([1,0,0,0,0,0],'P0','xx')
###


# SVD

from scipy.linalg import sqrtm
funcs_ri=[np.real,np.imag]

name='_equal'

def standarizeMom(mom):
    t=np.abs(mom)
    t.sort()
    return t

FFs=['A20','B20','C20']
@click.command()
@click.option('-e','--ens')
def run(ens):
    inpath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run3/{ens2full[ens]}/data_merge/dataJVR.h5'
    outpath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run3/{ens2full[ens]}/data_merge/dataJVR_SVD{name}.h5'
    with h5py.File(inpath) as f, h5py.File(outpath,'w') as fw:
        moms_N=f['N.h5/moms'][:]
        dic_N={}
        for i,mom in enumerate(moms_N):
            dic_N[tuple(mom)]=i
        cN=f['N.h5/data/N_N'][:]
        
        file='N-j.h5'
        moms_3pt=f[file]['moms'][:]
        keys=list(f['N-j.h5/data'].keys()); keys.sort()
        for ikey,key in enumerate(keys):
            # if ikey<294:
            #     continue
            j,tf=key.split('_')
            if j=='jv1':
                continue
            # if j!='jq;stout10':
            #     continue
            tf=int(tf)
            print(ens,f'{ikey}/{len(keys)}',key,end='              \r')
            data=f['N-j.h5/data'][key][:]
            
        
            for imom,mom in enumerate(moms_3pt):
                # if not np.all(mom==[0,0,0,0,0,0]):
                #     continue
                mom_str='_'.join([str(ele) for ele in mom])
                pa=mom[:3]; q=mom[3:6]; pb=pa+q
                
                cNa=cN[:,:,dic_N[tuple(standarizeMom(pa))]]
                cNb=cN[:,:,dic_N[tuple(standarizeMom(pb))]]
                Njk=len(cNa)
                
                L=ens2N[ens]
                n1vec=np.array(mom[:3]); nqvec=np.array(mom[3:6])
                nvec=n1vec+nqvec
                pvec=nvec*(2*np.pi/L); p1vec=n1vec*(2*np.pi/L)
                qvec=nqvec*(2*np.pi/L)
                
                xE_jk=np.sqrt(pvec.dot(pvec)+ens2mN[ens]**2)
                xE1_jk=np.sqrt(p1vec.dot(p1vec)+ens2mN[ens]**2)
                Q2_jk=(qvec.dot(qvec) - (xE_jk-xE1_jk)**2 )
                Q2=np.mean(Q2_jk)
                
                c3pt=data[:,:,imom,:,:]
                ratio=c3pt/np.sqrt(
                        cNa[:,tf:tf+1]*cNb[:,tf:tf+1]*\
                        cNa[:,:tf+1][:,::-1]/cNa[:,:tf+1]*\
                        cNb[:,:tf+1]/cNb[:,:tf+1][:,::-1]
                )[:,:,None,None]
                
                pirs=[(proj,insert,ri) for proj in projs for insert in inserts for ri in [0,1] if insert[0]==insert[1] and useQ(mom,proj,insert)[ri]]
                # pirs=[(proj,insert,ri) for proj in projs for insert in inserts for ri in [0,1] if useQ(mom,proj,insert)[ri]]
                G=np.array([[funcs_ri[ri](ME2FF(m,pvec,p1vec,proj,insert)) for proj,insert,ri in pirs] for m in ens2mN[ens]])
                
                if len(G[0])==0:
                    rank=0
                else:
                    U, S, VT = np.linalg.svd(G[0])
                    tol = 1e-10
                    rank = np.sum(S > tol)
                if rank == 3 or (rank==1 and np.all(q==[0,0,0])):
                    M_all=np.transpose([funcs_ri[ri](ratio[:,:,projs.index(proj),inserts.index(insert)]) for proj,insert,ri in pirs],[1,2,0])
                    t=np.zeros([Njk,tf+1,rank])
                    for tc in range(tf+1):
                        M=M_all[:,tc]
                        cov=yu.jackmec(M)[-1]
                        cov=np.diag(np.diag(cov))
                        covI=np.linalg.inv(cov)
                        covIsq = sqrtm(covI)
                        def get(g,m):
                            gt=covIsq@g
                            u,s,vT=np.linalg.svd(gt)
                            sI=np.zeros(gt.T.shape)
                            np.fill_diagonal(sI,1/s)
                            return vT.T@sI@(u.T)@covIsq@m
                        F=np.array([get(g[:,:rank],m) for g,m in zip(G,M)])
                        t[:,tc,:]=F
                    for i in range(t.shape[-1]):
                        FF=FFs[i]
                        key=f'{FF}_{j}/{mom_str}/{tf}'
                        if key in fw:
                            del fw[key]
                        fw.create_dataset(key,data=t[:,:,i])
        
# for ens in enss:
#     run(ens)
run()