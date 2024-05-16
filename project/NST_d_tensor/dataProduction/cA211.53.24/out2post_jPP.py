'''
cat data_aux/cfgs_run | xargs -n 1 -I @ -P 10 python3 -u out2post_jPP.py -c @ > log/out2post_jPP.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import aux

postcode='jPP-b-tensor'
def cfg2out(cfg):
    path='/project/s1174/lyan/code/scratch/run/NST_d/cA211.53.24/jPP/data_out/'+cfg+'/'
    return path

flags={
    'convert_gms_Yan':True, # adjust structure of gms
    'removeAdditionalDimension_P':True,
}


@click.command()
@click.option('-c','--cfg')
def run(cfg):
    assert(len(cfg[1:])==4 and cfg[1:].isdigit())
    inpath=cfg2out(cfg)
    outpath='data_post/'+cfg+'/'
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
                if not file.endswith('jPi.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    src='sx00sy00sz00st00'
                    moms_old=fr[src]['mvec'][:]
                    moms=opabsDic['post'][aux.set2key({'j','pib'})]
                    dic=aux.moms2dic(moms_old)
                    pi2s=[pi2 for pi2 in fr[src].keys() if pi2.startswith('pi2=') ]
                    dic_pi2={}
                    for i,pi2 in enumerate(pi2s):
                        dic_pi2[pi2]=i
                    jMap=[dic[tuple(mom[9:12])] for mom in moms]
                    pi2Map=[dic_pi2['pi2={}_{}_{}'.format(str(mom[0]),str(mom[1]),str(mom[2]))] for mom in moms]

                    fw.create_dataset('moms',data=moms)
                    fw.create_dataset('inserts',data=aux.gjList)
                            
                    for src in fr.keys():
                        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                        src_new='st{:03d}'.format(st)
                        
                        t=np.array([fr[src][pi2]['juu_pi0u'][:] for pi2 in pi2s])
                        tu=t[...,0]+1j*t[...,1]
                        t=np.array([fr[src][pi2]['jdd_pi0d'][:] for pi2 in pi2s])
                        td=t[...,0]+1j*t[...,1]
                        factor=(-1)*np.conj(1j/np.sqrt(2)) # (adj sgn) * (src pion phase)
                        for j in ['j+','j-']:
                            sgn=-1 if j=='j+' else 1
                            t=(tu+td*sgn)*factor # - for (ju +/- jd)(pi0u - pi0d)
                            if flags['removeAdditionalDimension_P']:
                                t=t[:,:,:,0]
                            t=t[pi2Map,:,jMap]
                            t=np.transpose(t,[1,0,2])
                            t=t[:,:,:len(aux.gjList)]
                            if flags['convert_gms_Yan']:
                                t_gmMap=np.array([0,1,2,3,4, 5,6,7,8,9, 11,12,10,13,14, 15])
                                t=t[...,t_gmMap]
                                t_gmMul=np.array([1,1,1,1,1, 1,1,1,1,1, 1j,-1j,1j,1j,1j, 1j])
                                t=t*t_gmMul[None,None,:]   
                            fw.create_dataset('data/'+src_new+'/'+j+'_pi0',data=t.astype('complex64'))
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
                    src='sx00sy00sz00st00'
                    moms_old=fr[src]['mvec'][:]
                    moms=opabsDic['post'][aux.set2key({'pia','pib'})]
                    dic=aux.moms2dic(moms_old)
                    pi2s=[pi2 for pi2 in fr[src].keys() if pi2.startswith('pi2=') ]
                    dic_pi2={}
                    for i,pi2 in enumerate(pi2s):
                        dic_pi2[pi2]=i
                    pf2Map=[dic[tuple(mom[6:9])] for mom in moms]
                    pi2Map=[dic_pi2['pi2={}_{}_{}'.format(str(mom[0]),str(mom[1]),str(mom[2]))] for mom in moms]

                    fw.create_dataset('moms',data=moms)
                            
                    for src in fr.keys():
                        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                        src_new='st{:03d}'.format(st)
                        
                        for ky in ['pi+_pi+','pi-_pi-']:
                            t=np.array([fr[src][pi2][ky][:] for pi2 in pi2s])
                            t=t[...,0]+1j*t[...,1]
                            if flags['removeAdditionalDimension_P']:
                                t=t[:,:,:,0]
                            t=t[pi2Map,:,pf2Map,0].T
                            t=(-1)*t # -1 for g5-conj sign of source pi
                            fw.create_dataset('data/'+src_new+'/'+ky,data=t.astype('complex64'))
                        
                        for ky in ['pi0_pi0']:
                            t=np.array([fr[src][pi2]['pi0u_pi0u'][:] for pi2 in pi2s])
                            tuu=t[...,0]+1j*t[...,1]
                            t=np.array([fr[src][pi2]['pi0d_pi0d'][:] for pi2 in pi2s])
                            tdd=t[...,0]+1j*t[...,1]
                            t=(-1)*(tuu+tdd)/2 # 2 for |1j/sqrt(2)|^2 of pi, -1 for g5-conj sign of source pi
                            if flags['removeAdditionalDimension_P']:
                                t=t[:,:,:,0]
                            t=t[pi2Map,:,pf2Map,0].T
                            fw.create_dataset('data/'+src_new+'/'+ky,data=t.astype('complex64'))
                        
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()