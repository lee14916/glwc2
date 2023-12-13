import os,sys
import h5py  
import numpy as np
import math,cmath
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
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

ens='cA211.530.24'
lat=yu.LatticeEnsemble(ens)


path='/project/s1174/lyan/code/projectData/NST_a/test/data_merge/NST_a.h5_all'
data=yu.load(path)
Ncfg=yu.deepKey(data['2pt'],2).shape[0]
print('Ncfg='+str(Ncfg))


import pickle
with open('aux/avgDirection.pkl','rb') as f:
    avgD=pickle.load(f)

def rotate(opabj,rot):
    temp=[] # rotate insert
    for coe,opab,insert in opabj:
        gm,j,tf=insert.split('_')
        if gm in ['id','gt','g5','g5gt']:
            temp.append((coe,opab,insert))
        elif gm in ['gx','gy','gz']:
            i_gm={'gx':0,'gy':1,'gz':2}[gm]
            for j_gm,val in enumerate(avgD[rot]['gamma_i'][i_gm,:]):
                if np.abs(val)<1e-7:
                    continue
                insert_new='_'.join([['gx','gy','gz'][j_gm],j,tf])
                temp.append((coe*val,opab,insert_new))
        elif gm in ['g5gx','g5gy','g5gz']:
            i_gm={'g5gx':0,'g5gy':1,'g5gz':2}[gm]
            for j_gm,val in enumerate(avgD[rot]['gamma_i'][i_gm,:]):
                if np.abs(val)<1e-7:
                    continue
                insert_new='_'.join([['g5gx','g5gy','g5gz'][j_gm],j,tf])
                temp.append((coe*val,opab,insert_new))
        else:
            1/0
            
    temp2=[] # rotate opa
    for coe,opab,insert in temp:
        opa,opb=opab.split('_')
        g,pt,irrep,occ,lam,fla=opa.split(';')
        assert(pt in ['0,0,0','0,0,1'])
        if pt !='0,0,0':
            opa_new=';'.join([g,rot,irrep,occ,lam,fla])
            temp2.append((coe,'_'.join([opa_new,opb]),insert))
        else:
            i_lam={'l1':0,'l2':1}[lam]
            for j_lam,val in enumerate(avgD[rot]['irrep_row'][i_lam,:]):
                if np.abs(val)<1e-7:
                    continue
                opa_new=';'.join([g,pt,irrep,occ,['l1','l2'][j_lam],fla])
                temp2.append((coe*val,'_'.join([opa_new,opb]),insert))
                
    temp3=[] # rotate opb
    for coe,opab,insert in temp2:
        opa,opb=opab.split('_')
        g,pt,irrep,occ,lam,fla=opb.split(';')
        assert(pt in ['0,0,0','0,0,1'])
        if pt !='0,0,0':
            opb_new=';'.join([g,rot,irrep,occ,lam,fla])
            temp3.append((coe,'_'.join([opa,opb_new]),insert))
        else:
            i_lam={'l1':0,'l2':1}[lam]
            for j_lam,val in enumerate(avgD[rot]['irrep_row'][i_lam,:]):
                if np.abs(val)<1e-7:
                    continue
                opb_new=';'.join([g,pt,irrep,occ,['l1','l2'][j_lam],fla])
                temp3.append((coe*np.conj(val),'_'.join([opa,opb_new]),insert))

    return temp3

def getDat(opabj,diag):
    return np.sum([data['3pt'][opab][insert][diag]*coe  for coe,opab,insert in opabj],axis=0)


thre=4

opabs=[]
for i_opab,opab in enumerate(data['3pt'].keys()):
    opa,opb=opab.split('_')
    _,pta,_,_,_,_=opa.split(';'); _,ptb,_,_,_,_=opb.split(';')
    if (pta not in ['0,0,0','0,0,1']) or (ptb not in ['0,0,0','0,0,1']):
        continue
    opabs.append(opab)

for i_opab,opab in enumerate(opabs):
    opa,opb=opab.split('_')
    _,pta,_,_,_,_=opa.split(';'); _,ptb,_,_,_,_=opb.split(';')
    if (pta not in ['0,0,0','0,0,1']) or (ptb not in ['0,0,0','0,0,1']):
        continue
    for i_insert,insert in enumerate(data['3pt'][opab].keys()):
        gm,j,tf=insert.split('_')
        if j!='j+' or tf!='10':
            continue    
        opabj=[(1,opab,insert)]
        for diag in data['3pt'][opab][insert].keys():
            for rot in ['0,0,-1','0,1,0','0,-1,0','1,0,0','-1,0,0']:
                t_dat=getDat(opabj,diag)
                t_dat2=getDat(rotate(opabj,rot),diag)
                for npRI in [np.real,np.imag]:
                    mean,err,_=yu.jackknife(npRI(t_dat))
                    mean2,err2,_=yu.jackknife(npRI(t_dat2))

                    t=np.abs((mean[0]-mean2[0])/np.sqrt(err[0]**2+err2[0]**2)); s=np.sum([1 if ele>thre else 0 for ele in t])
                    t2=np.abs((mean[0]-mean2[0])/np.sqrt(err[0]**2+err[0]**2)); s2=np.sum([1 if ele>thre else 0 for ele in t2])
                    t3=np.abs((mean[0]-mean2[0])/np.sqrt(err2[0]**2+err2[0]**2)); s3=np.sum([1 if ele>thre else 0 for ele in t3])
                    if s>0.5 or s2>0.5 or s3>0.5: 
                        print(opabj,diag,rot,npRI)
                        print(rotate(opabj,rot))
                        print(s,s2,s3)
                        print(t)
                        print(t2)
                        print(t3)
        print(i_opab,len(opabs),i_insert,len(data['3pt'][opab].keys()))
                        
    # break