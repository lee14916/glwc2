
'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u out2post_NJN_booster.py -c @ > log/out2post_NJN_booster.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import auxx as aux

postcode='NJN-booster'
def cfg2out(cfg):
    path=f'/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/01_Nsgm/cB211.072.64_base/data_NJN_from_booster/{cfg}/'
    return path 

def get_phase(src_int,mom):
    (sx,sy,sz,st)=src_int
    return np.exp(1j*(2*np.pi/aux.lat_L)*(np.array([sx,sy,sz])@mom))

assert(len(aux.tfList)==1)
assert(aux.tfList[0] in [12,14,16,18,20])

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=cfg2out(cfg)
    outpath=aux.pathBaseTf+'data_post/'+cfg+'/'
    files = os.listdir(inpath)
    os.makedirs(outpath,exist_ok=True)
    
    outfile=outpath+'NJN.h5_'+postcode
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with open(aux.path_opabsDic,'rb') as f:
            opabsDic=pickle.load(f)
        flag_setup=True
        with h5py.File(outfile,'w') as fw:
            for file in files:
                if file not in ['NJN.h5']:
                    continue
                with h5py.File(f'{inpath}{file}') as f:
                    if flag_setup:
                        moms_old=[list(mom) for mom in f['moms'][:]]
                        moms=opabsDic['post'][aux.set2key({'N','j'})]
                        assert(np.all(moms==[[0,0,0, 0,0,0, 0,0,0, 0,0,0]]))
                        momMap=[moms_old.index([0,0,0])]           
                        fw.create_dataset('moms',data=moms)
                        fw.create_dataset('inserts',data=aux.gjList)
                        flag_setup=False
                    for tf in aux.tfList:
                        tfStr=str(tf)
                        for src in f[f'dt{tf}/up/Local/P4'].keys():
                            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                            src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                            t=f[f'dt{tf}/up/Local/P4/{src}'][:]+f[f'dt{tf}/dn/Local/P4/{src}']
                            t=t[:,momMap,:,None]
                            t=t[...,[0,0,0,0]]
                            t=np.transpose(t,[0,1,3,2])
                            t[:,:,1:]*=9999
                            fw.create_dataset('data/'+src_new+'/p_'+'j+'+'_p_deltat_'+tfStr,data=t.astype('complex64'))
                            t*=9999
                            fw.create_dataset('data/'+src_new+'/p_'+'j-'+'_p_deltat_'+tfStr,data=t.astype('complex64'))
                    
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()