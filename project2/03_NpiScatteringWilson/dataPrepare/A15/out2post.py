'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u out2post.py -c @ > log/out2post.out & 
'''
import os, click, h5py, re, pickle
import numpy as np

postcode='0mom'
runPath='/capstor/store/cscs/userlab/s1174/lyan/code/scratch/run/03_NpiScatteringWilson/A15/'
basePath='/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/03_NpiScatteringWilson/A15/'

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'{runPath}out/{cfg}/'
    outpath=f'{basePath}data_post/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    
    files=[file for file in os.listdir(inpath) if file.endswith('.h5')]
    
    outfile=outpath+'N.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw:
            flag_moms=False
            for file in files:
                if not file.endswith('N.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    for src in fr.keys():
                        if not flag_moms:
                            moms=fr[src]['mvec'][:]
                            flag_moms=True
                            fw.create_dataset('moms',data=moms)
                            fw.create_dataset('notes',data=['time,mom,dirac'])
                        assert(np.all(moms==fr[src]['mvec'][:]))
                        
                        for tp in fr[src].keys():
                            if tp in ['mvec']:
                                continue
                            t=fr[f'{src}/{tp}'][:,:,0]
                            t=t[...,0]+1j*t[...,1]
                            fw.create_dataset(f'data/{src}/{tp}',data=t)
        os.remove(outfile_flag)
                            
                            
    outfile=outpath+'BWZ.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw:
            flag_moms=False
            for file in files:
                if not file.endswith('NpiScatteringWilson.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    if not flag_moms:
                        moms=fr['moms'][:]
                        flag_moms=True
                        fw.create_dataset('moms',data=moms)
                        fw.create_dataset('notes',data=fr['notes'][:])
                    assert(np.all(moms==fr['moms'][:]))
                    for src in fr['data'].keys():
                        for tp in fr['data'][src].keys():
                            t=fr[f'data/{src}/{tp}'][:]
                            fw.create_dataset(f'data/{src}/{tp}',data=t)
        os.remove(outfile_flag)
        
    print('flag_cfg_done: '+cfg)
             
run()