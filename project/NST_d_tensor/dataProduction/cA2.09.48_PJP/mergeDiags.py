'''
nohup python3 -u mergeDiags.py > log/mergeDiags.out & 
'''
import os, click, h5py, re
import numpy as np
import aux

mergecode='NST_d.h5_N-PJP'

@click.command()
def run():
    inpath='data_merge/temp2/'
    outfile='data_merge/'+mergecode
        
    diags=[diag.strip('.h5') for diag in os.listdir(inpath)]

    with open(aux.path_cfgs) as f:
        cfgs=f.read().splitlines()
    
    with h5py.File(outfile,'w') as fw:
        for diag in diags:
            infile=inpath+diag+'.h5'
            with h5py.File(infile) as fr:
                for ky in ['cfgs']:
                    if ky not in fw.keys():
                        fw.copy(fr[ky],fw,name=ky)
                for ky in ['opabs','inserts','data','srcs']:
                    if ky in fr.keys():
                        fw.copy(fr[ky],fw,name='diags/'+diag+'/'+ky)

    print('flag_done: '+mergecode)
                
run()