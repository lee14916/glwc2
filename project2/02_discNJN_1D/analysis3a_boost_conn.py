'''
nohup python3 -u analysis3a_boost_conn.py > log/analysis3a_boost_conn.out &
'''

import os,sys,warnings
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

# mpl.rcParams.update({"axes.grid" : True})
import util as yu
yu.flag_fast=False

enss=['b','c','d','e']
enss=['b','c','d']
ens2full={'a24':'cA211.53.24','a':'cA2.09.48','b':'cB211.072.64','c':'cC211.060.80','d':'cD211.054.96','e':'cE211.044.112'}
ens2label={'a24':'A24','a':'A48','b':'B64','c':'C80','d':'D96','e':'E112'}
ens2a={'a24':0.0908,'a':0.0938,'b':0.07957,'c':0.06821,'d':0.05692,'e':0.04892} # fm
ens2N={'a24':24,'a':48,'b':64,'c':80,'d':96,'e':112}
ens2N_T={'a24':24*2,'a':48*2,'b':64*2,'c':80*2,'d':96*2,'e':112*2}

hbarc = 1/197.3
ens2aInv={ens:1/(ens2a[ens]*hbarc) for ens in enss} # MeV

projs=['P0','Px','Py','Pz']
inserts=["tt", "tx", "ty", "tz", "xx", "xy", "xz", "yy", "yz", "zz"]
xyztdic={'x':0,'y':1,'z':2,'t':3}

cfg2old=lambda cfg: cfg[1:]+'_r'+{'a':'0','b':'1','c':'2','d':'3'}[cfg[0]]
cfg2new=lambda cfg: {'0':'a','1':'b','2':'c','3':'d'}[cfg[-1]] + cfg[:4]

path='/p/project1/ngff/li47/code/glwc2/project2/02_discNJN_1D/dataPrepare/cB211.072.64/data_aux/cfgs_run'
with open(path,'r') as f:
    cfgs_run=f.read().splitlines()
    
path='/p/scratch/ngff/kummer3/runs/Preprocessed_files/threeps/'
cfgs_conn=[cfg2new(cfg[:7]) for cfg in os.listdir(path)]
cfgs_conn.sort()

cfgs=list(set(cfgs_run).intersection(set(cfgs_conn)))
cfgs.sort()
len(cfgs)

tfs=[8,10,12,14]


data={}
path='/p/scratch/ngff/kummer3/runs/Preprocessed_files/threeps/'
setupQ=True
for icfg,cfg in enumerate(cfgs):
    print(f'{icfg}/{len(cfgs)}',end='              \r')
    with h5py.File(f'{path}/{cfg2old(cfg)}.h5') as f:
        for tf_key in f.keys():
            tf=int(tf_key[-2:])
            if setupQ:
                p1s_keys=list(f[tf_key]['up'].keys()); p1s_keys.sort()
                p1s=[[int(p1key[2:4]),int(p1key[6:8]),int(p1key[10:12])] for p1key in p1s_keys]
                
                p1key2imom={}
                for p1key in p1s_keys:
                    moms=[list(mom) for mom in f[tf_key]['up'][p1key]['P4_P']['OneD']['mvec'][:]]
                    imom=moms.index([0,0,0])
                    p1key2imom[p1key]=imom
                
                projs_key=['P4_P','P4G5G1_P','P4G5G2_P','P4G5G3_P']
                
                tfs=[int(tf[-2:]) for tf in f.keys()]
                data={f'{j}_{tf}':[] for tf in tfs for j in ['j+;conn','j-;conn']}
                
                setupQ=False
            
            tu=np.transpose([[f[tf_key]['up'][p1key][projkey]['OneD']['threep'][:,p1key2imom[p1key]] for p1key in p1s_keys] for projkey in projs_key],[2,1,0,3,4,5])
            td=np.transpose([[f[tf_key]['dn'][p1key][projkey]['OneD']['threep'][:,p1key2imom[p1key]] for p1key in p1s_keys] for projkey in projs_key],[2,1,0,3,4,5]) 
            tp=tu+td; tm=tu-td
            
            def func(t):
                t=t[...,0]+1j*t[...,1]
                t=t[...,[1,2,3,4]]
                t=(t + np.transpose(t,[0,1,2,4,3]))/2
                t=t-np.eye(4)[None,None,None,:,:]*np.trace(t,axis1=3,axis2=4)[:,:,:,None,None]/4
                t=np.transpose([t[:,:,:,xyztdic[insert[0]],xyztdic[insert[1]]] for insert in inserts],[1,2,3,0])
                t[:,:,1:,:]*=-1
                return t
            
            tp=func(tp); tm=func(tm)
            data[f'j+;conn_{tf}'].append(tp)
            data[f'j-;conn_{tf}'].append(tm)
            
    # break

for key in data.keys():
    data[key]=np.array(data[key])

path='/p/project1/ngff/li47/code/projectData/02_discNJN_1D/data_conn/B64_2units_conn.h5'
with h5py.File(path,'w') as f:
    f.create_dataset('cfgs',data=cfgs)
    f.create_dataset('moms',data=p1s)
    for key in data.keys():
        f.create_dataset(f'data/{key}',data=data[key])   
