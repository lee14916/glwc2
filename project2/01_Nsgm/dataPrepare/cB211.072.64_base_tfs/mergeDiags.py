'''
nohup python3 -u mergeDiags.py > log/mergeDiags.out & 
'''
import os, click, h5py, re
import numpy as np
import auxx as aux

mergecode='Nsgm.h5'

@click.command()
def run():
    inpath=f'{aux.pathBaseTf}data_merge/temp2/'
    outfile=f'{aux.pathBaseTf}data_merge/'+mergecode
        
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
                        
        for diag in ['j','pi0f']:
            flag_setup=True
            t_dat={}
            for cfg in cfgs:
                infile=aux.cfg2post(cfg,diag)
                with h5py.File(infile) as fr:
                    if flag_setup:
                        if 'inserts' in fr.keys():
                            fw.copy(fr['inserts'],fw,name='VEV/'+diag+'/inserts')
                        moms=fr['moms']
                        i_mom0=None
                        for i,mom in enumerate(moms):
                            if tuple(mom)==(0,)*12:
                                i_mom0=i
                                break
                        assert(i_mom0!=None)
                        flag_setup=False
                    for fla in fr['data'].keys():
                        if fla not in t_dat:
                            t_dat[fla]=[]
                        t=fr['data'][fla][:]
                        t=np.mean(t[:,i_mom0],axis=0)
                        t_dat[fla].append(t)
            for fla in t_dat.keys():
                fw.create_dataset('VEV/'+diag+'/data/'+fla,data=np.array(t_dat[fla]).astype('complex64'))

    print('flag_done: '+mergecode)
                
run()