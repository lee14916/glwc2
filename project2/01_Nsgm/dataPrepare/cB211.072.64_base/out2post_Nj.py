'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u out2post_Nj.py -c @ > log/out2post_Nj.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import auxx as aux

postcode='Nj'
def cfg2out(cfg):
    path=f'/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/01_Nsgm/cB211.072.64_base/data_Nj_from_booster/{cfg}/'
    return path

flags={
    'g5H':True,
}

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    assert(len(cfg[1:])==4 and cfg[1:].isdigit())
    inpath=cfg2out(cfg)
    outpath=aux.pathBase+'data_post/'+cfg+'/'
    files = os.listdir(inpath)
    os.makedirs(outpath,exist_ok=True)
    
    with open(aux.path_opabsDic,'rb') as f:
        opabsDic=pickle.load(f)
    
    outfile=outpath+'N.h5_'+postcode; outfile_bw=outpath+'N_bw.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        for case in ['fw','bw']:
            t_outfile=outfile if case=='fw' else outfile_bw
            with h5py.File(t_outfile,'w') as fw:
                moms=opabsDic['post'][aux.set2key({'N'})]
                fw.create_dataset('moms',data=moms)
                
                for file in files:
                    if not file.startswith('N.h5'):
                        continue
                    infile=inpath+file
                    with h5py.File(infile) as fr:
                        moms_old=fr['moms'][:]
                        if len(moms_old.shape)==1:
                            moms_old=np.array([moms_old])
                        dic=aux.moms2dic(moms_old)
                        mom_flip = -1 if case == 'bw' else 1
                        momMap=[dic[tuple(mom[3:6])] for mom in mom_flip*np.array(moms)]                           
                        for src in fr['data'].keys():
                            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                            src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                            
                            ky2new={'N1_N1':'p_p','N2_N2':'n_n'}
                            for ky,ky_new in ky2new.items():
                                t=fr['data'][src][ky][:]
                                if case=='bw':
                                    t=fr['data_bw'][src][ky][:]
                                    t=-np.roll(np.flip(t,axis=0),1,axis=0)
                                    t[0]*=0
                                t=t[:aux.Tpack,:,:]
                                t=t[:,momMap]
                                fw.create_dataset('data/'+src_new+'/'+ky_new,data=t.astype('complex64'))
        os.remove(outfile_flag)

    outfile=outpath+'j.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if not file.endswith('j.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    moms_old=fr['moms'][:]
                    moms=opabsDic['post'][aux.set2key({'j'})]
                    dic=aux.moms2dic(moms_old)
                    momMap=[dic[tuple(mom[9:12])] for mom in moms]
                    
                    assert(np.all([insert.decode() for insert in fr['inserts']]==aux.gjList))
                    fw.create_dataset('moms',data=moms)
                    fw.create_dataset('inserts',data=aux.gjList)

                    for j in ['j+','j-','js','jc']:
                        t=fr[f'data/{j}'][:,momMap]
                        if flags['g5H']:
                            sgn={'j+':1,'j-':-1,'js':1,'jc':1}[j]
                            g5Cj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':-1,'g5':1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}
                            sgnConj=np.array([g5Cj[gj] for gj in aux.gjList])
                            momMap_neg=[dic[tuple(-np.array(mom[-3:]))] for mom in moms]
                            t2=fr[f'data/{j}'][:,momMap_neg]
                            t=(t+sgn*np.conj(t2)*sgnConj[None,None,:])/2
                        fw.create_dataset(f'data/{j}',data=t.astype('complex128'))
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()