'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u out2post_jPP.py -c @ > log/out2post_jPP.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import auxx as aux

postcode='jPP'
def cfg2out(cfg):
    path='/capstor/store/cscs/userlab/s1174/lyan/code/scratch/run/Nsgm/cB211.072.64/jPP/out/'+cfg+'/'
    return path

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    assert(len(cfg[1:])==4 and cfg[1:].isdigit())
    inpath=cfg2out(cfg)
    outpath=aux.pathBase+'data_post/'+cfg+'/'
    files = os.listdir(inpath)
    os.makedirs(outpath,exist_ok=True)

    outfile=outpath+'jPi.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with open(aux.path_opabsDic,'rb') as f:
            opabsDic=pickle.load(f)
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if not file.endswith('jPP.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    ky0='data'
                    src='st000'
                    moms_old=fr['moms'][:]
                    moms=opabsDic['post'][aux.set2key({'j','pib'})]
                    dic=aux.moms2dic(moms_old)
                    fw.create_dataset('moms',data=moms)
                    fw.create_dataset('inserts',data=aux.gjList)
                    
                    i_Gi=0
                    assert(fr['Gis'][i_Gi].decode()=='id')
                    sgn_Gi=aux.g5Cj[fr['Gis'][i_Gi].decode()]
                    Gfs= fr['Gfs'][:]
                    sgns_Gf=np.array([aux.g5Cj[Gf.decode()] for Gf in Gfs])
                            
                    for src in fr[ky0].keys():
                        tu=fr[ky0][src]['uBuL_uBuS'][:,:,:,i_Gi]
                        td=fr[ky0][src]['dBdL_dBdS'][:,:,:,i_Gi]
                        
                        tuFlip=np.conj(fr[ky0][src]['uBuL_uBuS'][:,:,:,i_Gi]) * sgns_Gf[None,None,:] * sgn_Gi
                        tdFlip=np.conj(fr[ky0][src]['dBdL_dBdS'][:,:,:,i_Gi]) * sgns_Gf[None,None,:] * sgn_Gi
                        
                        tu=np.transpose([ tu[:,dic[tuple(np.concatenate([mom[0:3],mom[9:12]]))],:] if tuple(mom[0:3]+mom[9:12]) in dic \
                            else tuFlip[:,dic[tuple(-np.concatenate([mom[0:3],mom[9:12]]))],:]  for mom in moms], [1,0,2])
                        td=np.transpose([ td[:,dic[tuple(np.concatenate([mom[0:3],mom[9:12]]))],:] if tuple(mom[0:3]+mom[9:12]) in dic \
                            else tdFlip[:,dic[tuple(-np.concatenate([mom[0:3],mom[9:12]]))],:]  for mom in moms], [1,0,2])
                        
                        factor=1/np.sqrt(2) * aux.gtCj[fr['Gis'][i_Gi].decode()] # (src sigma phase)
                        for j in ['j+','j-']:
                            sgn=1 if j=='j+' else -1
                            t=(tu+td*sgn)*factor
                            fw.create_dataset('data/'+src+'/'+j+'_sgm',data=t.astype('complex64'))
        os.remove(outfile_flag)
        
    outfile=outpath+'P.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with open(aux.path_opabsDic,'rb') as f:
            opabsDic=pickle.load(f)
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if not file.endswith('P.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    ky0='data'
                    src='sx00sy00sz00st00'
                    moms_old=fr['moms'][:]
                    moms=opabsDic['post'][aux.set2key({'pia','pib'})]
                    dic=aux.moms2dic(moms_old)
                    fw.create_dataset('moms',data=moms)
                    
                    i_Gi=0
                    assert(fr['Gis'][i_Gi].decode()=='id')
                    i_Gf=0
                    assert(fr['Gfs'][i_Gf].decode()=='id')
                    sgn_Gi=aux.g5Cj[fr['Gis'][i_Gi].decode()]
                    sgn_Gf=aux.g5Cj[fr['Gfs'][i_Gf].decode()]
                            
                    for src in fr[ky0].keys():
                        tu=fr[ky0][src]['uBuS_uBuS'][:,:,i_Gf,i_Gi]
                        td=fr[ky0][src]['dBdS_dBdS'][:,:,i_Gf,i_Gi]
                        
                        tuFlip=np.conj(fr[ky0][src]['uBuS_uBuS'][:,:,i_Gf,i_Gi]) * sgn_Gf * sgn_Gi
                        tdFlip=np.conj(fr[ky0][src]['dBdS_dBdS'][:,:,i_Gf,i_Gi]) * sgn_Gf * sgn_Gi
                        
                        tu=np.transpose([ tu[:,dic[tuple(np.concatenate([mom[0:3],mom[9:12]]))]] if tuple(mom[0:3]+mom[9:12]) in dic \
                            else tuFlip[:,dic[tuple(-np.concatenate([mom[0:3],mom[9:12]]))]]  for mom in moms], [1,0])
                        td=np.transpose([ td[:,dic[tuple(np.concatenate([mom[0:3],mom[9:12]]))]] if tuple(mom[0:3]+mom[9:12]) in dic \
                            else tdFlip[:,dic[tuple(-np.concatenate([mom[0:3],mom[9:12]]))]]  for mom in moms], [1,0])
                        
                        t=(tu+td)/2 * aux.gtCj[fr['Gis'][i_Gi].decode()]  # 2 for |1/sqrt(2)|^2 of sigma
                        ky='sgm_sgm'
                        fw.create_dataset('data/'+src+'/'+ky,data=t.astype('complex64'))
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()