# cyclone
'''
cat data_aux/cfgs_all_jg | xargs -I @ -P 10 python3 -u raw2post_jg.py -c @ > log/raw2post_jg.out & 
'''
import re, click
import h5py, os
import numpy as np

import timeit

ens='cD211.054.96'
stouts=range(40+1)
basePath='/onyx/qdata/gspanoudes/Local_operators/Matrix_elements/cD96/data/gluon_loops_stout/'
replicas=['cD96a','cD96b','cD96c','cD96d']; labels=['a','b','c','d']

beta={'cB211.072.64':1.778,'cC211.060.80':1.836,'cD211.054.96':1.900}[ens]
factor=beta/3

inserts=["tt", "tx", "ty", "tz", "xx", "xy", "xz", "yy", "yz", "zz"]
inserts_nums=["33", "03", "13", "23", "00", "01", "02", "11", "12", "22"] # symmetric 

replica2label={replica:label for replica,label in zip(replicas,labels)}
label2replica={label:replica for replica,label in zip(replicas,labels)}
cfg2old=lambda cfg: cfg[1:]+'_r'+{'a':'0','b':'1','c':'2','d':'3'}[cfg[0]]
cfg2new=lambda cfg: {'0':'a','1':'b','2':'c','3':'d'}[cfg[-1]] + cfg[:4]

# Nmax={'cB211.072.64':16,'cC211.060.80':16,'cD211.054.96':16}[ens]
Nmax={'cB211.072.64':4,'cC211.060.80':4,'cD211.054.96':16}[ens]
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
base_momList=[[x,y,z] for x in t_range for y in t_range for z in t_range if np.linalg.norm([x,y,z])**2<=Nmax+0.1]
base_momList.sort()

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    outpath=f'/nvme/h/cy22yl1/projectData/02_discNJN_1D/{ens}/data_post/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    with h5py.File(outpath+'jg.h5','w') as fw:
        flag=True
        for stout in stouts:
            inpath=f'{basePath}stout{stout}/{label2replica[cfg[0]]}/{cfg[1:]}/'
            t_jg=[]
            for i_n in inserts_nums:
                file=inpath+f'gLoops_ultralocal_Clv_EMT_{i_n}_{cfg[1:]}.dat'
                with open(file,'r') as f:
                    t=f.read().split('\n')[:-1]
                if flag:
                    tt=[row.split(' ') for row in t]
                    N_tot=len(t)
                    N_T=np.max([int(row[1]) for row in tt])+1
                    N_mom=N_tot//N_T
                    
                    moms=[[int(ele) for ele in row[3:6]] for row in tt[:N_mom]]
                    
                    dic={}
                    for i,mom in enumerate(moms):
                        dic[tuple(mom)]=i
                    inds_mom=[dic[tuple(mom)] for mom in base_momList]
                    
                    fw.create_dataset('moms',data=base_momList)
                    fw.create_dataset('inserts',data=inserts)
                    
                    flag=False
                
                t=[float(part[0])+1j*float(part[1]) for part in (row.split(' ')[-2:] for row in t)]
                t=np.reshape(t,(N_T,N_mom))
                t_jg.append(t)
            t_jg=np.transpose(t_jg,[1,2,0])*factor
            t_jg=t_jg[:,inds_mom]
            fw.create_dataset(f'data/jg;stout{stout}',data=t_jg)
            
    print('flag_cfg_done: '+cfg)
            
run()