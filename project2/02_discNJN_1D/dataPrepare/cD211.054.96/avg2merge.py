'''
nohup python3 -u avg2merge.py > log/avg2merge.out & 
'''
import os, click, h5py, re
import numpy as np

ens='cD211.054.96'

path='data_aux/cfgs_run'
with open(path,'r') as f:
    cfgs=f.read().splitlines()
    
basepath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run2/{ens}/'
inpath_loop=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post/'

def run():
    os.makedirs(f'{basepath}data_merge',exist_ok=True)
    outpath=f'{basepath}data_merge/data.h5'
    
    dat={}
    for cfg in cfgs:
        print(cfg,end='                \r')
        inpath=f'{basepath}data_avgmore/{cfg}/'
        for file in os.listdir(inpath):
            infile=f'{inpath}{file}'
            if file not in dat:
                dat[file]={}
            with h5py.File(infile) as f:
                for key in f.keys():
                    if key in ['data']:
                        continue
                    if key not in dat[file]:
                        dat[file][key]=f[key][:]
                        
                for fla in f['data'].keys():
                    key=f'data/{fla}'
                    if key not in dat[file]:
                        dat[file][key]=[]
                    dat[file][key].append(f[key][:])
                    
        path=inpath_loop+cfg+'/j.h5'
        file='j.h5'; 
        if file not in dat.keys():
            dat[file]={}
        with h5py.File(path) as f:
            moms=[tuple(mom) for mom in f['moms'][:]]
            i_mom=moms.index((0,0,0))
            dat[file]['inserts']=f['inserts;g{m,Dn};tl'][:]
            
            keys=[key for key in f['data'].keys() if key.endswith(';g{m,Dn};tl') or key.startswith('jg')]
            for key in keys:
                t=np.mean(f['data'][key][:,i_mom,:],axis=0)
                key_new=f'{key}_vev'
                if key_new not in dat[file]:
                    dat[file][key_new]=[]
                dat[file][key_new].append(t)
        # break
                
    with h5py.File(outpath,'w') as f:
        for file in dat.keys():
            for key in dat[file].keys():
                f.create_dataset(f'{file}/{key}',data=dat[file][key])
    
    
run()