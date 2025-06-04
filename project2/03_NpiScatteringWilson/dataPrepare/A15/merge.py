'''
nohup python3 -u merge.py > log/merge.out & 
'''
import os, click, h5py, re
import numpy as np

mergecode='NpiScatteringWilson.h5'

basePath='/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/03_NpiScatteringWilson/A15/'

@click.command()
def run():
    inpath=f'{basePath}data_avgsrc/'
    outpath=f'{basePath}data_merge/'
    os.makedirs(outpath,exist_ok=True)
    
    path='data_aux/cfgs_run'
    with open(path,'r') as f:
        cfgs=f.read().splitlines()
    
    with h5py.File(f'{outpath}{mergecode}','w') as fw:
        fw.create_dataset('cfgs',data=cfgs)
        cfg=cfgs[0]
        infile=f'{inpath}{cfg}/NPBWZM.h5'
        with h5py.File(infile) as f:
            for ky in f.keys():
                if ky not in ['data']:
                    fw.create_dataset(ky,data=f[ky][:])
            conts=list(f['data'].keys())
            

        data={cont:[] for cont in conts}
        for cfg in cfgs:
            infile=f'{inpath}{cfg}/NPBWZM.h5'
            with h5py.File(infile) as f:
                for cont in conts:
                    data[cont].append(f['data'][cont][:])
        for cont in conts:
            fw.create_dataset(f'data/{cont}',data=data[cont])
        

    
run()