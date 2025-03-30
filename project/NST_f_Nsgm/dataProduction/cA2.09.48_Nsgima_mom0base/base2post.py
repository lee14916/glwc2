'''
cat data_aux/cfgs_run | xargs -n 1 -I @ -P 10 python3 -u base2post.py -c @ > log/base2post.out &
'''
import os, click, h5py, re, pickle
import numpy as np
import aux

with open(aux.path_opabsDic,'rb') as f:
    opabsDic=pickle.load(f)

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'/capstor/store/cscs/userlab/s1174/lyan/code/projectData/NST_f/cA2.09.48_Nsigma/data_post/{cfg}/'
    outpath=f'data_post/{cfg}/'
    
    os.makedirs(outpath,exist_ok=True)
    
    for file in os.listdir(inpath):
        infile=inpath+file
        outfile=outpath+file
        
        if file.split('.h5')[0] in ['B3pt','W3pt','Z3pt','NJN']:
            continue
        
        with h5py.File(outfile,'w') as fw, h5py.File(infile) as fr:
            moms_old=fr['moms'][:]
            moms=np.array(opabsDic['post'][aux.set2key(aux.diag2dgtp[file.split('.h5')[0]])])
            dic=aux.moms2dic(moms_old)
            moms_map=[dic[tuple(mom)] for mom in moms]
            
            fw.create_dataset('moms',data=moms)
            if 'inserts' in fr.keys():
                fw.create_dataset('inserts',data=fr['inserts'])
            
            if file.split('.h5')[0] not in ['j','pi0f']:
                for src in fr['data'].keys():
                    for fla in fr[f'data/{src}'].keys():
                        fw.create_dataset(f'data/{src}/{fla}',data=fr[f'data/{src}/{fla}'][:,moms_map])
            else:
                for fla in fr['data'].keys():
                    fw.create_dataset(f'data/{fla}',data=fr[f'data/{fla}'][:,moms_map])
                    
    print('flag_cfg_done: '+cfg)
    
run()