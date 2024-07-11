'''
cat data_aux/cfgs_run | xargs -n 1 -I @ -P 10 python3 -u out2post_PJP.py -c @ > log/out2post_PJP.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import aux

postcode='PJP-a'
def cfg2out(cfg):
    path='/p/project/pines/li47/code/scratch/run/NST_d/cA211.53.24/PJP/data_out/'+cfg+'/'
    return path


flags={
    'convert_gms':True, # adjust structure of gms
}

tfs=[10,12,14]

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    assert(len(cfg[1:])==4 and cfg[1:].isdigit())
    inpath=cfg2out(cfg)
    outpath='data_post/'+cfg+'/'
    files = os.listdir(inpath)
    os.makedirs(outpath,exist_ok=True)

    outfile=outpath+'PJP.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with open(aux.path_opabsDic,'rb') as f:
            opabsDic=pickle.load(f)
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if not file.endswith('PJP.h5'):
                    continue
                infile=inpath+file
                with h5py.File(infile) as fr:
                    src=list(fr.keys())[0]
                    moms_old=fr[src]['mvec'][:]
                    moms=opabsDic['post'][aux.set2key({'pia','j','pib'})]
                    dic=aux.moms2dic(moms_old)
                    j_map=[dic[tuple(mom[9:12])] for mom in moms]
                    
                    dt0='dt{}'.format(tfs[0])
                    pf2s=[ele for ele in fr[src][dt0].keys() if ele.startswith('pf2=') ]
                    pi2s=[ele for ele in fr[src][dt0][pf2s[0]].keys() if ele.startswith('pi2=') ]
                    assert(len(pf2s)==len(pi2s))
                    
                    flas=list(fr[src][dt0][pf2s[0]][pi2s[0]].keys())
                    dic_pf2={}
                    for i,pf2 in enumerate(pf2s):
                        x,y,z=pf2[4:].split('_')
                        x,y,z=int(x),int(y),int(z)
                        dic_pf2[(x,y,z)]=i
                    pf2_map=[dic_pf2[tuple(mom[6:9])] for mom in moms]
                    
                    dic_pi2={}
                    for i,pi2 in enumerate(pi2s):
                        x,y,z=pi2[4:].split('_')
                        x,y,z=int(x),int(y),int(z)
                        dic_pi2[(x,y,z)]=i
                    pi2_map=[dic_pi2[tuple(mom[:3])] for mom in moms]

                    fw.create_dataset('moms',data=moms)
                    fw.create_dataset('inserts',data=aux.gjList)
                    
                    for src in fr.keys():
                        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                        src_new='st{:03d}'.format(st)
                        
                        for tf in tfs: 
                            t_dic={}
                            for fla in flas:
                                t=np.array([[fr[src]['dt{}'.format(tf)][pf2][pi2][fla][:] for pf2 in pf2s] for pi2 in pi2s])
                                t=t[...,0]+1j*t[...,1]
                                t=t[pi2_map,pf2_map,:,j_map]
                                if flags['convert_gms']:
                                    t_gmMap=np.array([0,1,2,3,4, 5,6,7,8,9, 11,12,10,13,14, 15])
                                    t=t[...,t_gmMap]
                                    t_gmMul=np.array([1,1,1,1,1, 1,1,1,1,1, 1j,-1j,1j,1j,1j, 1j])
                                    t=t*t_gmMul[None,None,:]
                                t_dic[fla]=np.transpose(t,[1,0,2])   
                            
                            for pi in ['pi+','pi-']:
                                for j in ['j+','j-']:
                                    key=pi+'_'+j+'_'+pi
                                    sgn={'j+':+1,'j-':-1}[j]
                                    t=t_dic[pi+'_ju_'+pi]+sgn*t_dic[pi+'_jd_'+pi]
                                    fw.create_dataset('data/'+src_new+'/'+key+'_deltat_'+str(tf),data=t.astype('complex64'))

                            pi='pi0'
                            for j in ['j+','j-']:
                                key=pi+'_'+j+'_'+pi
                                sgn={'j+':+1,'j-':-1}[j]
                                t=(t_dic['pi0u_ju_pi0u1']+t_dic['pi0u_ju_pi0u2']) + sgn* (t_dic['pi0d_jd_pi0d1']+t_dic['pi0d_jd_pi0d2'])
                                t=t/2
                                fw.create_dataset('data/'+src_new+'/'+key+'_deltat_'+str(tf),data=t.astype('complex64'))

        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()