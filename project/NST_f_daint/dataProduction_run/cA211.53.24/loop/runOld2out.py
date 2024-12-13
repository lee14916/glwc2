'''
cat runAux/cfgs_runt | xargs -n 1 -I @ -P 10 python3 -u runOld2out.py -c @ > log/runOld2out.out & 
'''
import os, click, h5py, re, pickle
import numpy as np

lat_L,lat_T=24,48

Gs=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'/project/s1174/lyan/code/scratch/run/NST_d/cA211.53.24/loops/data_out/{cfg}/'
    outpath=f'out/{cfg}/'
    os.makedirs(outpath,exist_ok=True)

    infile=inpath+f'Diagram{cfg[1:]}_insertLoop.h5'
    outfile=outpath+f'loopL.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw, h5py.File(infile) as f:
            st='sx00sy00sz00st00'
            ps=f[st]['mvec'][:]
            
            moms=[]; moms_map=[]
            for i,mom in enumerate(ps):
                mom2=mom[0]**2+mom[1]**2+mom[2]**2
                if mom2<=1:
                    moms.append(list(mom))
                    moms_map.append(i)
                    
            fw.create_dataset('moms',data=moms)
            fw.create_dataset('t,mom,G',data=[])
            fw.create_dataset('Gs',data=Gs)
            
            t_gmMap=np.array([0,1,2,3,4, 5,6,7,8,9, 11,12,10,13,14, 15])
            sign_g=np.array([1, 1,1,1,1, 1, 1,1,1,1, 1j,-1j,1j, 1j,1j,1j])
            
            for sid in f[st].keys():
                if sid == 'mvec':
                    continue
                assert(sid.split('_')[-1]=='1')
                seed='seed='+sid.split('id')[0][4:]
                id='id='+sid.split('_')[0].split('id')[1]
                
                t=f[st][sid]['up'][:]
                t=t[...,0]+1j*t[...,1]
                t=np.complex128(t)
                t=t[:,moms_map]
                t=t[...,t_gmMap]
                t=t*sign_g[None,None,:]   
                            
                fw.create_dataset('data/'+seed+'/'+id+'/up',data=t)
            
        os.remove(outfile_flag)
        
    print('flag_cfg_done: '+cfg)
    
run()