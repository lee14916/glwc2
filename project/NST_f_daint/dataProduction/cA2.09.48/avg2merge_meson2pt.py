'''
nohup python3 -u avg2merge_meson2pt.py > log/avg2merge_meson2pt.out & 
'''
import os, click, h5py, re, pickle
import numpy as np


mergecode='NST_f_meson2pt.h5'

inpath='data_avgsrc/'
inpath='data_avgmore/'

# import aux
path_cfgs='data_aux/cfgs_run'
# assert(aux.path_cfgs==path_cfgs)

@click.command()
def run():
    outpath='data_merge/'
    os.makedirs(outpath,exist_ok=True)
    outfile=outpath+mergecode
    
    with open(path_cfgs) as f:
        cfgs=f.read().splitlines()
    cfgs.sort()
    
    with h5py.File(outfile,'w') as fw:
        fw.create_dataset('cfgs',data=cfgs)
        for diag in ['P','pi0f-pi0i']:
            dat={}; datVEV={}
            for cfg in cfgs:
                with h5py.File(inpath+f'{cfg}/{diag}.h5') as fr:
                    if cfg==cfgs[0]:
                        fw.create_dataset(f'diags/{diag}/moms',data=fr['moms'][:])
                    for fla in fr['data'].keys():
                        if fla not in dat:
                            dat[fla]=[]
                        dat[fla].append(fr['data'][fla][:])
                    
                    if 'VEV' in fr.keys():
                        for fla in fr['VEV'].keys():
                            if fla not in datVEV:
                                datVEV[fla]=[]
                            datVEV[fla].append(fr['VEV'][fla][()])
                    
            for fla in dat.keys():
                fw.create_dataset(f'diags/{diag}/data/{fla}',data=dat[fla])  
            for fla in datVEV.keys():
                fw.create_dataset(f'VEV/pi0f/data/{fla}',data=datVEV[fla])        
                
    print('flag_done: '+mergecode)
        
run()