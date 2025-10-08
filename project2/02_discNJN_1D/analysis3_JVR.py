'''
nohup python3 -u analysis3_JVR.py -e b > log/analysis3_JVR_b.out &
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



# JVR = Jacknife + Vaccum subtraction + Renormalization

path='data_aux/RCs.pkl'
with open(path,'rb') as f:
    ens2RCs_me=pickle.load(f)
ens2RCs={ens:{} for ens in enss}
for ens in enss:
    for key in ens2RCs_me[ens]:
        if key.endswith('err'):
            continue
        ens2RCs[ens][key]=yu.jackknife_pseudo([ens2RCs_me[ens][key]],np.array([[ens2RCs_me[ens][f'{key}_err']**2+1e-10]]),ens2Njk[ens])[:,0]

j2j1={
    'jq':[[1,'j+'],[1,'js'],[1,'jc']],
    'jv1':[[1,'j-']],
    'jv2':[[1,'j+'],[-2,'js']],
    'jv3':[[1,'j+'],[1,'js'],[-3,'jc']]
}

def standarizeMom(mom):
    t=np.abs(mom)
    t.sort()
    return t

@click.command()
@click.option('-e','--ens')
def run(ens):
    inpath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run3/{ens2full[ens]}/data_merge/data.h5'
    outpath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run3/{ens2full[ens]}/data_merge/dataJVR.h5'
    with h5py.File(inpath) as f, h5py.File(outpath,'w') as fw:
        moms_N=f['N.h5/moms'][:]
        dic_N={}
        for i,mom in enumerate(moms_N):
            dic_N[tuple(mom)]=i
        cN=f['N.h5/data/N_N'][:]
        cN=yu.jackknife(cN)
        fw.create_dataset(f'N.h5/data/N_N',data=cN)
        
        for key in f[f'N.h5'].keys():
            if key=='data':
                continue
            fw.copy(f['N.h5'][key],fw,name=f'N.h5/{key}')
        
        for key in f[f'discNJN_j+;g{{m,Dn}};tl.h5'].keys():
            if key=='data':
                continue
            fw.copy(f[f'discNJN_j+;g{{m,Dn}};tl.h5'][key],fw,name=f'N-j.h5/{key.split(";")[0]}' if key.startswith('inserts') else f'N-j.h5/{key}')
        
        tfs=[int(key.split('_')[-1]) for key in f['discNJN_j+;g{m,Dn};tl.h5/data'].keys()]
        tfs.sort()
        
        for tf in tfs:
            dic={}
            for j in ['j+','js','jc'] + [f'jg;stout{stout}' for stout in stouts]:
                print(ens,tf,j,end='                   \r')
                file=f'discNJN_{j};g{{m,Dn}};tl.h5' if not j.startswith('jg') else f'discNJN_jg;stout.h5'
                fla_base=f'N_N_{j};g{{m,Dn}};tl_' if not j.startswith('jg') else f'N_N_{j}_'
                
                moms_3pt=f[file]['moms'][:]
                inds_0mom=[i for i,mom in enumerate(moms_3pt) if np.all(mom[3:]==[0,0,0])]
                inds_N_0mom=[dic_N[tuple(standarizeMom(mom[:3]))] for i,mom in enumerate(moms_3pt) if np.all(mom[3:]==[0,0,0])]
                
                t=f[file]['data'][f'{fla_base}{tf}'][:]
                t=yu.jackknife(t)
                
                key=f'{j};g{{m,Dn}};tl_vev' if not j.startswith('jg') else f'{j}_vev'
                tvev=f['j.h5'][key][:,:]
                tvev=yu.jackknife(tvev)
                tvev_tt=tvev[:,inserts.index('tt')]
                inds_xyz=[inserts.index(insert) for insert in ['xx','yy','zz']]
                tvev_zz=np.mean(tvev[:,inds_xyz],axis=1)
                
                ind_P0=projs.index('P0')
                ind=inserts.index('tt')
                t[:,:,inds_0mom,ind_P0,ind] -= cN[:,tf:tf+1,inds_N_0mom] * tvev_tt[:,None,None]
                for insert in ['xx','yy','zz']:
                    ind=inserts.index(insert)
                    t[:,:,inds_0mom,ind_P0,ind] -= cN[:,tf:tf+1,inds_N_0mom] * tvev_zz[:,None,None]
                
                dic[j]=t
            
            for j in ['jv2','jv3']: # jv1=0
                t=0
                for factor,j1 in j2j1[j]:
                    t += factor * dic[j1]
                Z=np.transpose([ens2RCs[ens][f'Zqq(mu=nu)'] if insert[0]==insert[1] else ens2RCs[ens][f'Zqq(mu!=nu)'] for i,insert in enumerate(inserts)])
                
                t=t*Z[:,None,None,None,:]
                key=f'N-j.h5/data/{j}_{tf}'
                if key in fw:
                    del fw[key]
                fw.create_dataset(key,data=t)

            t=0
            for factor,j1 in j2j1['jq']:
                t += factor * dic[j1]
            tq_bare=t
            Zqq=np.transpose([ens2RCs[ens][f'Zqq^s(mu=nu)'] if insert[0]==insert[1] else ens2RCs[ens][f'Zqq^s(mu!=nu)'] for i,insert in enumerate(inserts)])
            for stout in stouts:
                tg_bare=dic[f'jg;stout{stout}']
                Zgg=np.transpose([ens2RCs[ens][f'Zgg^{stout}(mu=nu)'] if insert[0]==insert[1] else ens2RCs[ens][f'Zgg^{stout}(mu!=nu)'] for i,insert in enumerate(inserts)])
                Zgq=np.transpose([ens2RCs[ens][f'Zgq(mu=nu)'] if insert[0]==insert[1] else ens2RCs[ens][f'Zgq(mu!=nu)'] for i,insert in enumerate(inserts)])
                Zqg=np.transpose([ens2RCs[ens][f'Zqg(mu=nu)'] if insert[0]==insert[1] else ens2RCs[ens][f'Zqg(mu!=nu)'] for i,insert in enumerate(inserts)])
                
                tq = tq_bare*Zqq[:,None,None,None,:] + tg_bare*Zqg[:,None,None,None,:]
                tg = tq_bare*Zgq[:,None,None,None,:] + tg_bare*Zgg[:,None,None,None,:]
                
                key=f'N-j.h5/data/jq;stout{stout}_{tf}'
                if key in fw:
                    del fw[key]
                fw.create_dataset(key,data=tq)
                key=f'N-j.h5/data/jg;stout{stout}_{tf}'
                if key in fw:
                    del fw[key]
                fw.create_dataset(key,data=tg)

run()