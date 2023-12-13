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

path='/project/s1174/lyan/code/projectData/NST_a/test/data_merge/NST_a.h5_avgD'
data=yu.load(path)
Ncfg=yu.deepKey(data['2pt'],2).shape[0]
print('Ncfg='+str(Ncfg))

path='/project/s1174/lyan/code/projectData/NST_a/test/data_merge/NST_a.h5_all'
data_all=yu.load(path)
Ncfg=yu.deepKey(data_all['2pt'],2).shape[0]
print('Ncfg='+str(Ncfg))

thre=3

for i,opab in enumerate(data['3pt'].keys()):
    print(i,len(data['3pt'].keys()),end='       \r')
    for insert in data['3pt'][opab].keys():
        gm,j,tf=insert.split('_')
        if j!='j+' or tf!='10':
            continue
        for diag in data['3pt'][opab][insert].keys():
            t_dat=data['3pt'][opab][insert][diag]
            t_dat2=data_all['3pt'][opab][insert][diag]
            for npRI in [np.real,np.imag]:
                mean,err,_=yu.jackknife(npRI(t_dat))
                mean2,err2,_=yu.jackknife(npRI(t_dat2))

                t=np.abs((mean[0]-mean2[0])/np.sqrt(err[0]**2+err2[0]**2)); s=np.sum([1 if ele>thre else 0 for ele in t])
                if s>0.5: 
                    print(opab,diag,npRI)
                    print(s)
                    print(t)