'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u out2post_loops.py -c @ > log/out2post_loops.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import auxx as aux

postcode='Nstoc25'
def cfg2out(cfg):
    path='/capstor/store/cscs/userlab/s1174/lyan/code/scratch/run/Nsgm/cB211.072.64/loop/out/'+cfg+'/'
    return path

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    assert(len(cfg[1:])==4 and cfg[1:].isdigit())
    inpath=cfg2out(cfg)
    outpath=aux.pathBase+'data_post/'+cfg+'/'
    files = os.listdir(inpath)
    os.makedirs(outpath,exist_ok=True)

    # outfile=outpath+'j.h5_'+postcode
    # outfile_flag=outfile+'_flag'
    # if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
    #     with open(outfile_flag,'w') as f:
    #         pass
    #     with open(aux.path_opabsDic,'rb') as f:
    #         opabsDic=pickle.load(f)
    #     with h5py.File(outfile,'w') as fw:
    #         for file in files:
    #             if not file.endswith('loopL.h5'):
    #                 continue
    #             infile=inpath+file
    #             with h5py.File(infile) as fr:
    #                 ky0='data'
    #                 moms_old=fr['moms'][:]
    #                 moms=opabsDic['post'][aux.set2key({'j'})]
    #                 dic=aux.moms2dic(moms_old)
    #                 momMap=[dic[tuple(mom[9:12])] for mom in moms]
    #                 momMap_neg=[dic[tuple(-np.array(mom)[9:12])] for mom in moms]

    #                 fw.create_dataset('moms',data=moms)
    #                 fw.create_dataset('inserts',data=aux.gjList)
                
    #                 Nstoc=0; data={'j+':0,'j-':0}
    #                 for stoc in fr[ky0].keys():
    #                     assert(stoc.startswith('seed='))
    #                     for id in fr[ky0][stoc].keys():
    #                         assert(id.startswith('id='))
    #                         flas=fr[ky0][stoc][id].keys()
    #                         assert(len(flas)==1)
    #                         fla=list(flas)[0]
    #                         assert(fla in ['up','dn'])

    #                         t_stoc=1
    #                         Nstoc+=t_stoc
    #                         t=fr[ky0][stoc][id][fla][:]
    #                         t=t[:,:,:len(aux.gjList)]

    #                         gList=aux.gjList
    #                         sgnConj=np.array([aux.g5Cj[gj] for gj in gList])
    #                         if fla=='up':
    #                             t_up=t[:,momMap]
    #                             t_dn=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
    #                         elif fla=='dn':
    #                             t_up=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
    #                             t_dn=t[:,momMap]
                            
    #                         data['j+']+=t_stoc*(t_up+t_dn)
    #                         data['j-']+=t_stoc*(t_up-t_dn)
                    
    #                 for j in ['j+','j-']:
    #                     t=data[j]/Nstoc
    #                     fw.create_dataset('data/'+'/'+j,data=t.astype('complex128'))

    #     os.remove(outfile_flag)
        
    outfile=outpath+'pi0f.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with open(aux.path_opabsDic,'rb') as f:
            opabsDic=pickle.load(f)
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if not file.endswith('loopS.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    ky0='data'
                    moms_old=fr['moms'][:]
                    moms=opabsDic['post'][aux.set2key({'pia'})]
                    dic=aux.moms2dic(moms_old)
                    momMap=[dic[tuple(mom[6:9])] for mom in moms]
                    momMap_neg=[dic[tuple(-np.array(mom)[6:9])] for mom in moms]
                    fw.create_dataset('moms',data=moms)
                
                    Nstoc=0; data=0
                    for stoc in fr[ky0].keys():
                        assert(stoc.startswith('seed='))
                        for id in fr[ky0][stoc].keys():
                            assert(id.startswith('id='))
                            flas=fr[ky0][stoc][id].keys()
                            assert(len(flas)==1)
                            fla=list(flas)[0]
                            assert(fla in ['up','dn'])

                            t_stoc=1
                            Nstoc+=t_stoc
                            t=fr[ky0][stoc][id][fla][:]

                            gList=['id']
                            sgnConj=np.array([aux.g5Cj[gj] for gj in gList])
                            if fla=='up':
                                t_up=t[:,momMap]
                                t_dn=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
                            elif fla=='dn':
                                t_up=np.conj(t[:,momMap_neg])*sgnConj[None,None,:]
                                t_dn=t[:,momMap]
                                
                            data += t_stoc * np.array([t_up+t_dn,t_up-t_dn])

                    data=data/Nstoc
                    t=data[0,:,:,0] # first 0 for u+d, last 0 for gm=id
                    t=t*1/np.sqrt(2)
                    
                    fw.create_dataset('data/'+'/sgm',data=t.astype('complex128'))
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()