'''
cat data_aux/cfgs_run | xargs -n 1 -I @ -P 10 python3 -u out2post_loops.py -c @ > log/out2post_loops.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import aux

postcode='loops-b-Nstoc200-tensor'
def cfg2out(cfg):
    path='/project/s1174/lyan/code/scratch/run/NST_d/cA211.53.24/loops/data_out/'+cfg+'/'
    return path

flags={
    'convert_gms_Yan':True, # adjust structure of gms
}

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    assert(len(cfg[1:])==4 and cfg[1:].isdigit())
    inpath=cfg2out(cfg)
    outpath='data_post/'+cfg+'/'
    files = os.listdir(inpath)
    os.makedirs(outpath,exist_ok=True)

    outfile=outpath+'j.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with open(aux.path_opabsDic,'rb') as f:
            opabsDic=pickle.load(f)
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if not file.endswith('insertLoop.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    src='sx00sy00sz00st00'
                    moms_old=fr[src]['mvec'][:]
                    moms=opabsDic['post'][aux.set2key({'j'})]
                    dic=aux.moms2dic(moms_old)
                    momMap=[dic[tuple(mom[9:12])] for mom in moms]
                    momMap_neg=[dic[tuple(-np.array(mom)[9:12])] for mom in moms]

                    fw.create_dataset('moms',data=moms)
                    fw.create_dataset('inserts',data=aux.gjList)
                
                    Nstoc=0; data={'j+':0,'j-':0}
                    for stoc in fr[src].keys():
                        if not stoc.startswith('stoc'):
                            continue
                        flas=fr[src][stoc].keys()
                        assert(len(flas)==1)
                        fla=list(flas)[0]
                        assert(fla in ['up','dn'])

                        t_stoc=int(stoc.split('_')[1])
                        Nstoc+=t_stoc
                        t=fr[src][stoc][fla][:]
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,:,:len(aux.gjList)]

                        gList=aux.gjList
                        sgnConj=np.array([aux.g5Cj[gj] for gj in gList])
                        if fla=='up':
                            t_up=t[:,momMap]
                            t_dn=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
                        elif fla=='dn':
                            t_up=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
                            t_dn=t[:,momMap]
                        
                        data['j+']+=t_stoc*(t_up+t_dn)
                        data['j-']+=t_stoc*(t_up-t_dn)
                    
                    for j in ['j+','j-']:
                        t=data[j]/Nstoc
                        if flags['convert_gms_Yan']:
                            t_gmMap=np.array([0,1,2,3,4, 5,6,7,8,9, 11,12,10,13,14, 15])
                            t=t[...,t_gmMap]
                            t_gmMul=np.array([1,1,1,1,1, 1,1,1,1,1, 1j,-1j,1j,1j,1j, 1j])
                            t=t*t_gmMul[None,None,:]    
                        fw.create_dataset('data/'+'/'+j,data=t.astype('complex64'))
        os.remove(outfile_flag)
        
    outfile=outpath+'pi0f.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with open(aux.path_opabsDic,'rb') as f:
            opabsDic=pickle.load(f)
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if not file.endswith('pi0Loop.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    src='sx00sy00sz00st00'
                    moms_old=fr[src]['mvec'][:]
                    moms=opabsDic['post'][aux.set2key({'pia'})]
                    dic=aux.moms2dic(moms_old)
                    momMap=[dic[tuple(mom[6:9])] for mom in moms]
                    momMap_neg=[dic[tuple(-np.array(mom)[6:9])] for mom in moms]
                    fw.create_dataset('moms',data=moms)
                
                    Nstoc=0; data=0
                    for stoc in fr[src].keys():
                        if not stoc.startswith('stoc'):
                            continue
                        flas=fr[src][stoc].keys()
                        assert(len(flas)==1)
                        fla=list(flas)[0]
                        assert(fla in ['up','dn'])

                        t_stoc=int(stoc.split('_')[1])
                        Nstoc+=t_stoc
                        t=fr[src][stoc][fla][:]
                        t=t[...,0]+1j*t[...,1]

                        gList=['g5']
                        sgnConj=np.array([aux.g5Cj[gj] for gj in gList])
                        if fla=='up':
                            t_up=t[:,momMap]
                            t_dn=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
                        elif fla=='dn':
                            t_up=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
                            t_dn=t[:,momMap]
                            
                        data += t_stoc * np.array([t_up+t_dn,t_up-t_dn])

                    data=data/Nstoc
                    t=data[1,:,:,0]
                    t=t*1j/np.sqrt(2)
                    
                    fw.create_dataset('data/'+'/pi0',data=t.astype('complex64'))
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()