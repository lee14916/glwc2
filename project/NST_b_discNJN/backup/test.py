import os,sys
import h5py  
import numpy as np
import math,cmath
import pickle
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('default')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.figsize'] = [6.4*1.2,4.8*1.2]
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['lines.marker'] = 's'
mpl.rcParams['lines.linestyle'] = ''
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['errorbar.capsize'] = 12
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['legend.fontsize'] = 24

mpl.rcParams.update({"axes.grid" : True})
import util as yu


path='/p/project/pines/li47/code/projectData/NST_b-discNJN/data_merge/NST_b.h5_main'
data=yu.load(path)
for ens in yu.ensembles:
    Ncfg=yu.deepKey(data[ens]['2pt'],2).shape[0]
    print(ens+': Ncfg='+str(Ncfg))
    
    
def getdat(ens,ins,tfs,case):
    flags={
        'cc2pt':False,
        'cc3pt':False
    }
    spins=[0,1]; flas=['N1','N2']; diags=['N','N_bw']
    # spins=[0,1]; flas=['N1']; diags=['N']
    diags_3pt=[diag+'-j' for diag in diags]
    t=np.mean([data[ens]['2pt']['t;0,0,0;{};{}_t;0,0,0;{};{}'.format(spin,fla,spin,fla)][diag] for spin in spins for fla in flas for diag in diags],axis=0)
    if flags['cc2pt']:
        t=(t+np.conj(t))/2
    t_2pta=t_2ptb=t
    
    cg,j=ins.split('_')
    gm={'gS':'id','gA':'g5gz','gT':'sgmxy'}[cg]
    pol={'gS':1,'gA':-1,'gT':-1}[cg]
    factor={'gS':1,'gA':1j,'gT':-1j}[cg]

    t_3pt={}
    for tf in tfs:
        insert='_'.join([gm,j,str(tf)])
        t_conn=data[ens]['3pt']['0mom'][ins+'_'+str(tf)]['NJN']
        fla2sgn={'N1':1,'N2':(-1 if j=='j-' else 1)*{'id':1,'g5gz':1,'sgmxy':1}[gm]}
        t_disc=np.mean([(
            data[ens]['3pt']['t;0,0,0;0;{}_t;0,0,0;0;{}'.format(fla,fla)][insert][diag]+
            data[ens]['3pt']['t;0,0,0;1;{}_t;0,0,0;1;{}'.format(fla,fla)][insert][diag]*pol
        )/2*fla2sgn[fla]*factor
                        for fla in flas for diag in diags_3pt],axis=0)
        t_3pt[tf]={'conn':t_conn,'disc':t_disc,'conn+disc':t_conn+t_disc}[case]
        if flags['cc3pt']:
            t_3pt[tf]= (t_3pt[tf] + np.conj(t_3pt[tf][:,::-1])*yu.gtCj[gm]*(np.conj(factor)/factor))/2
        
        t_VEV=data[ens]['VEV']['j']['_'.join([gm,j])]
    
    t=[[t_3pt[tf] for tf in tfs],[t_2pta,t_2ptb],[t_VEV]]
    return t


# tfmax check
thred=0.2

ens2tfmax={}
for irow,ens in enumerate(yu.ensembles):
    ins='gS_j+'
    tfs=[int(insert.split('_')[-1]) for insert in data[ens]['3pt']['0mom'].keys() if insert.startswith(ins)]; tfs.sort()
    dat=getdat(ens,ins,tfs,'conn')
    
    def func(dat):
        t=yu.meanDeep(dat)
        return np.real(t[1][0])
    
    (mean,err,cov)=yu.jackknife(dat,func)
    errR=err/np.abs(mean)
    for tf,ele in enumerate(errR[0]):
        if ele>thred:
            ens2tfmax[ens]=tf-1
            break
print(ens2tfmax)
ens2tfmin_best={'cB211.072.64': 8, 'cC211.060.80': 10, 'cD211.054.96':12}


ens2info={
    'cB211.072.64':
        {
            'a':0.07957,
            'datmin':range(1,9+1,1),
            'best':[6,6,6],
            'mN':0.38008987581293957,
            },
    'cC211.060.80':
        {
            'a':0.06821,
            'datmin':range(1,8+1,1),
            'best':[6,6,6],
            'mN':0.3227500301787349,
            },
    'cD211.054.96':
        {
            'a':0.05692,
            'datmin':range(1,7+1,1),
            'best':[6,6,6],
            'mN':0.27230159934716447,
            },
}

ins2ylims={
    'gS_j+':([2,14],[-1.5,3]),
    'gS_j-':([0.25,2],[-1,0.2]),
    'gA_j+':([0.5,0.7],[-0.5,0.5]),
    'gA_j-':([1,1.4],[-0.03,0.03]),
    'gT_j+':([0.5,0.8],[-0.05,0]),
    'gT_j-':([0.7,1.3],[-0.3,0.3]),
}

func_C2pt_2st=lambda t,E0,c0,dE1,rc1: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t))
func_C3pt_2st=lambda tf,tc,E0,a00,dE1,ra01,ra11: a00*np.exp(-E0*tf)*(1 + ra01*(np.exp(-dE1*tc) + np.exp(-dE1*(tf-tc))) + ra11*np.exp(-dE1*tf))

def run(ins,fast=False,reset=False):
    if reset:
        yu.res_fit={}
    fig, axs = yu.getFigAxs(len(yu.ensembles)+1,3+3)
    fig.suptitle(ins)
    yu.addRowHeader(axs,[ens[:2]+ens[-2:] for ens in yu.ensembles]+['tmin'])
    yu.addColHeader(axs,['conn','','conn+disc','','disc'])
    
    fmts=['s','s','s']
    
    best={}
    for irow,ens in enumerate(yu.ensembles):
        t={
        'gS_j+':yu.ens2info[ens]['factor_gS'],
        'gS_j-':yu.ens2info[ens]['factor_gS'],
        'gA_j+':yu.ens2info[ens]['factor_gAs'],
        'gA_j-':yu.ens2info[ens]['factor_gAv'],
        'gT_j+':yu.ens2info[ens]['factor_gT'],
        'gT_j-':yu.ens2info[ens]['factor_gT'],
        }
        prefactor=t[ins] if ins in t else 1
        
        for icol,case in enumerate(['conn','conn+disc','disc']):    
            # if (irow,icol) not in [(2,2)]:
            #     continue
            icol*=2
            ylim=ins2ylims[ins][0] if 'conn' in case else ins2ylims[ins][1]
            needsVEV=True if case!='conn' and ins in ['gS_j+'] else False
            tfs=[int(insert.split('_')[-1]) for insert in data[ens]['3pt']['0mom'].keys() if insert.startswith(ins)]; tfs.sort()
            dat=getdat(ens,ins,tfs,case)
            
            # ratio plot
            def func(dat):
                t=yu.meanDeep(dat)
                t=[t[0][i][1:tf]/np.sqrt(
                    np.real(t[1][0][tf])*np.real(t[1][1][tf])*\
                    np.real(t[1][0][1:tf][::-1])/np.real(t[1][0][1:tf])*\
                    np.real(t[1][1][1:tf])/np.real(t[1][1][1:tf][::-1])
                ) - (t[2][0] if needsVEV else 0) 
                for i,tf in enumerate(tfs)]
                # print(t)
                t=yu.prefactorDeep(t,prefactor)
                return t
            (mean,err,cov) = yu.jackknife(dat,func)
            for i_tf,tf in enumerate(tfs):
                tMean=mean[i_tf];tErr=err[i_tf]
                axs[irow,icol].errorbar(np.arange(1 - tf//2,tf//2),tMean,tErr,fmt=fmts[irow])
                axs[irow,icol].set_ylim(ylim)
                
            # initial fits
            datmin=ens2info[ens]['datmin'][0]
            datmin_3pt_tf=datmin_3pt_tci=datmin_3pt_tcf=datmin
            datmin_2pt=8
            tfs_2pt_fit=np.arange(datmin_2pt,ens2tfmax[ens]+1)
            tfs_3pt_fit=[tf for tf in tfs if tf>=datmin_3pt_tf and tf+1>(datmin_3pt_tci+datmin_3pt_tcf)]
            tcs_3pt_fit={tf:np.arange(datmin_3pt_tci,tf+1-datmin_3pt_tcf) for tf in tfs_3pt_fit}
            
            pars0_initial=[None,1e-8,0.5,1,2,1,1] # E0,c0,E1,rc1,g,ra01,ra11
            if case=='disc':
                pars0_initial=[None,1e-8,0.4,1,-0.2,1,1]
            pars0_initial[0]=ens2info[ens]['mN']
            pars0=pars0_initial[:4]
            tfs=np.arange(datmin_2pt,ens2tfmax[ens]+1)
            def func(dat):
                dat=dat[1][0][:,tfs]
                t=yu.meanDeep(dat)
                return np.real(t)
            def fitfunc(E0,c0,E1,rc1):
                return func_C2pt_2st(tfs,E0,c0,E1-E0,rc1)
            pars0_initial[:4]=yu.fit(dat,func,fitfunc,pars0=pars0,jk=False,label=';'.join([ins,ens,case,'meanOnly1']))[0]
            pars0_initial[0]=ens2info[ens]['mN']
            
            E0,c0,E1=pars0_initial[:3]
            pars0=pars0_initial[-3:]
            tfs=[int(insert.split('_')[-1]) for insert in data[ens]['3pt']['0mom'].keys() if insert.startswith(ins)]; tfs.sort()
            def func(dat):
                t=yu.meanDeep(dat)
                t_3pt=[np.real(prefactor*(t[0][i]- (t[1][0][tf]*t[2][0] if needsVEV else 0)))[tcs_3pt_fit[tf]] for i,tf in enumerate(tfs) if tf in tfs_3pt_fit]
                return t_3pt
            def fitfunc(g,ra01,ra11):
                t_3pt=[func_C3pt_2st(tf,tcs_3pt_fit[tf],E0,g*c0,E1-E0,ra01,ra11) for tf in tfs_3pt_fit]
                return t_3pt
            pars0_initial[-3:]=yu.fit(dat,func,fitfunc,pars0=pars0,jk=False,label=';'.join([ins,ens,case,'meanOnly2']))[0]
            pars0_initial[0]=ens2info[ens]['mN']
            
            pars0=pars0_initial
            fits=[]
            for datmin in ens2info[ens]['datmin']:
                print(ins,ens,case,datmin,end='                      \r')
                datmin_2pt=datmin_3pt_tf=datmin_3pt_tci=datmin_3pt_tcf=datmin
                datmin_2pt=ens2tfmin_best[ens]
                tfs_2pt_fit=np.arange(datmin_2pt,ens2tfmax[ens]+1)
                tfs_3pt_fit=[tf for tf in tfs if tf>=datmin_3pt_tf and tf+1>(datmin_3pt_tci+datmin_3pt_tcf)]
                tcs_3pt_fit={tf:np.arange(datmin_3pt_tci,tf+1-datmin_3pt_tcf) for tf in tfs_3pt_fit}
                Ny_2pt=len(tfs_2pt_fit); Ny_3pt=len([1 for tf in tfs_3pt_fit for tc in tcs_3pt_fit[tf]])
                Ny=Ny_2pt+Ny_3pt
                mask_cov=np.array([[1 if (i<Ny_2pt and j<Ny_2pt) or (i>=Ny_2pt and j>=Ny_2pt) else 0 for j in range(Ny)] for i in range(Ny)])
                
                def func(dat):
                    t=yu.meanDeep(dat)
                    t_2pt=np.real(t[1][0][tfs_2pt_fit])
                    t_3pt=[np.real(prefactor*(t[0][i]- (t[1][0][tf]*t[2][0] if needsVEV else 0)))[tcs_3pt_fit[tf]] for i,tf in enumerate(tfs) if tf in tfs_3pt_fit]
                    return [t_2pt]+t_3pt
                def fitfunc(E0,c0,E1,rc1,g,ra01,ra11):
                    t_2pt=func_C2pt_2st(tfs_2pt_fit,E0,c0,E1-E0,rc1)
                    t_3pt=[func_C3pt_2st(tf,tcs_3pt_fit[tf],E0,g*c0,E1-E0,ra01,ra11) for tf in tfs_3pt_fit]
                    return [t_2pt]+t_3pt
                pars_mean,pars_err,pars_cov,chi2R_mean,chi2R_err,Ndof=yu.fit(dat,func,fitfunc,mask_cov=None,pars0=pars0,jk=not fast,label=';'.join([ins,ens,case,str(datmin)]))
                fits.append((pars_mean,pars_err,chi2R_mean,Ndof))
                
                axs[irow,icol+1].errorbar([datmin],[pars_mean[4]],[pars_err[4]],fmt=fmts[irow],color='b',mfc='white')
                axs[irow,icol+1].annotate("%0.1f" %chi2R_mean,(datmin,pars_mean[4]-pars_err[4]-0.1),color='b')
                axs[irow,icol+1].set_ylim(ylim)
                
            # modelAvg
            pars_mean_MA,pars_err_MA,probs=yu.modelAvg(fits)
            # print(ens,case,'                     ')
            # print(probs)
            # print()
            
            t_mean=pars_mean_MA[4]; t_err=pars_err_MA[4]
            best[(ens,case)]=[ens2info[ens]['a']**2,t_mean,t_err]
            axs[irow,icol+1].fill_between([0,10],t_mean-t_err,t_mean+t_err,color='b',alpha=0.2)
    
    for icol,case in [(1,'conn'),(3,'conn+disc'),(5,'disc')]:
        # continue
        irow=len(yu.ensembles)
        a2s=np.array([best[(ens,case)][0] for ens in yu.ensembles])
        means=np.array([best[(ens,case)][1] for ens in yu.ensembles])
        errs=np.array([best[(ens,case)][2] for ens in yu.ensembles])
        axs[irow,icol].errorbar(a2s,means,errs,color='b',mfc='white')
        ylim=ins2ylims[ins][0] if 'conn' in case else ins2ylims[ins][1]
        axs[irow,icol].set_ylim(ylim)
        
        def fitfunc(a2,g0):
            return g0+0*a2
        popt,pcov=curve_fit(fitfunc,a2s,means,sigma=errs,absolute_sigma=True)
        r=fitfunc(a2s,*popt)-means
        # print(r)
        # print(errs)
        chi2R=r.T @ np.linalg.inv(np.diag(errs**2)) @ r / (3-1)
        a2s2=np.arange(0,0.008,0.001)
        res=fitfunc(a2s2,*popt)
        axs[irow,icol].plot(a2s2,res,'r-')
        axs[irow,icol].fill_between(a2s2,res-[np.sqrt(pcov[0,0])]*len(a2s2),res+[np.sqrt(pcov[0,0])]*len(a2s2),color='r',alpha=0.2)
        axs[irow,icol].errorbar([0],[popt[0]],np.sqrt(pcov[0,0]),color='r')
        axs[irow,icol].annotate("%0.1f" %chi2R,(0,popt[0]-np.sqrt(pcov[0,0])-0.1),color='r')
        
        def fitfunc(a2,g0,g1):
            return g0+g1*a2
        popt,pcov=curve_fit(fitfunc,a2s,means,sigma=errs,absolute_sigma=True)
        r=fitfunc(a2s,*popt)-means
        # print(r)
        # print(errs)
        chi2R=r.T @ np.linalg.inv(np.diag(errs**2)) @ r / (3-2)
        a2s2=np.arange(0,0.008,0.001)
        res=fitfunc(a2s2,*popt)
        es=np.array([np.sqrt(np.array([[1,a2]])@pcov@np.array([[1,a2]]).T)[0,0] for a2 in a2s2])
        axs[irow,icol].plot(a2s2,res,'b-')
        axs[irow,icol].fill_between(a2s2,res-es,res+es,color='b',alpha=0.2)
        axs[irow,icol].errorbar([0],[popt[0]],np.sqrt(pcov[0,0]),color='b')
        axs[irow,icol].annotate("%0.1f" %chi2R,(0,popt[0]-np.sqrt(pcov[0,0])-0.1),color='b')
            

for ins in ['gS_j+','gS_j-','gA_j+','gA_j-','gT_j+','gT_j-']:
    run(ins); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig('fig/'+ins+'.pdf')
    plt.close()
    
with open('dat/temp.pkl','wb') as f:
    pickle.dump(yu.res_fit,f)