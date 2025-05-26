'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u run2post_pion2pt.py -c @ > log/run2post_pion2pt.out & 
'''
import os, click, h5py, re, pickle
import numpy as np

postcode='0mom'
runPath='/capstor/store/cscs/userlab/s1174/lyan/code/scratch/run/03_NpiScatteringWilson/A15/'
basePath='/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/03_NpiScatteringWilson/A15/'

lat_L,lat_T=48,48

def key2mom(mom_key):
    t1,t2=mom_key.split('=')
    x,y,z=t2.split('_')
    x,y,z=int(x),int(y),int(z)
    return [x,y,z]

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'{runPath}run_pion2pt/{cfg}/'
    outpath=f'{basePath}data_post/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    
    files=[file for file in os.listdir(inpath) if file.endswith('.h5')]
    
    for file in files:
        if file.endswith('P.h5'):
            infile=f'{inpath}{file}'
            outfile=f'{outpath}P.h5_{postcode}'
            outfile_flag=outfile+'_flag'
            if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
                with open(outfile_flag,'w') as f:
                    pass
                with h5py.File(outfile,'w') as fw, h5py.File(infile) as f:
                    fw.create_dataset('notes',data=['tf,mom','mom=pi2,pf2'])
                    srcs=list(f.keys()); srcs.sort(); src=srcs[0]
                    pi2s_key=[pi2 for pi2 in f[f'{src}/PhiPhi'].keys() if pi2.startswith('pi2=')]; pi2s_key.sort(); pi2_key=pi2s_key[0]
                    pi2s=[key2mom(pi2) for pi2 in pi2s_key]
                    pf2s=[list(mom) for mom in np.atleast_2d(f[f'{src}/PhiPhi/mvec'])]
                    moms=[pi2+pf2 for pi2 in pi2s for pf2 in pf2s]
                    fw.create_dataset('moms',data=moms)
                    pi2Map=[pi2s.index(mom[:3]) for mom in moms]
                    pf2Map=[pf2s.index(mom[3:6]) for mom in moms]
                    
                    tps=list(f[f'{src}/PhiPhi/{pi2_key}'].keys()); tps.sort()
                    
                    for src in f.keys():
                        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                        src_new='st{:03d}'.format(st)
                        for tp in tps:
                            t=np.array([f[f'{src}/PhiPhi/{pi2_key}/{tp}'][:,:,0] for pi2_key in pi2s_key])
                            t=t[...,0]+1j*t[...,1]
                            t=t[pi2Map,:,pf2Map]
                            t=np.transpose(t,[1,0])
                            fw.create_dataset(f'data/{src_new}/{tp}',data=t)
                os.remove(outfile_flag)
        
    print('flag_cfg_done: '+cfg)
    
run()