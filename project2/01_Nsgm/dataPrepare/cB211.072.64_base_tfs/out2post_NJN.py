
'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u out2post_NJN.py -c @ > log/out2post_NJN.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import auxx as aux

postcode='NJN-Nsrc=1'
def cfg2out(cfg):
    t='' if cfg[0]=='a' else '_b'
    path = f'/capstor/store/cscs/userlab/s1174/fpittler/runs/njn_physical_point_64{t}/{cfg[1:]}/'
    return path 

flags={
    'flipNJNpc':True, # convention difference
    'flip_ab':True, # transpose the ab index
    'sign_pf1':True, # Tr[PC]=Pab Cba was used.
    'convert_gms':True, # adjust structure of gms
}

t_flipNJNpc= np.array([1,1,1, 1,1,1, 1,1,1, -1,-1,-1]) if flags['flipNJNpc'] else np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1])
t_flip_ab=[0,4,1,5] if flags['flip_ab'] else [0,1,4,5]

def get_phase(src_int,mom):
    (sx,sy,sz,st)=src_int
    return np.exp(1j*(2*np.pi/aux.lat_L)*(np.array([sx,sy,sz])@mom))

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=cfg2out(cfg)
    outpath=aux.pathBaseTf+'data_post/'+cfg+'/'
    files = os.listdir(inpath)
    os.makedirs(outpath,exist_ok=True)

    # outfile=outpath+'N.h5_'+postcode; outfile_bw=outpath+'N_bw.h5_'+postcode
    # outfile_flag=outfile+'_flag'
    # if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
    #     with open(outfile_flag,'w') as f:
    #         pass
    #     with open(aux.path_opabsDic,'rb') as f:
    #         opabsDic=pickle.load(f)
    #     for case in ['fw','bw']:
    #         t_outfile=outfile if case=='fw' else outfile_bw
    #         flag_setup=True
    #         with h5py.File(t_outfile,'w') as fw:
    #             for file in files:
    #                 if not file.endswith('N.h5'):
    #                     continue
    #                 infile=inpath+file
    #                 try:
    #                     with h5py.File(infile) as fr:
    #                         for src in fr.keys():
    #                             (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
    #                             (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
    #                             src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                                
    #                             if flag_setup:
    #                                 moms_old=fr[src]['mvec'][:]
    #                                 moms=opabsDic['post'][aux.set2key({'N'})]
    #                                 dic=aux.moms2dic(moms_old)
    #                                 mom_flip = -1 if case == 'bw' else 1
    #                                 momMap=[dic[tuple(mom[3:6])] for mom in mom_flip*np.array(moms)]                         
    #                                 fw.create_dataset('moms',data=moms)
    #                                 flag_setup=False
                                
    #                             ky2new={'NP':'p_p','N0':'n_n'}
    #                             for ky,ky_new in ky2new.items():
    #                                 t=fr[src][ky][:]
    #                                 t=t[...,0]+1j*t[...,1]
    #                                 if case=='bw':
    #                                     t=-np.roll(np.flip(t,axis=0),1,axis=0)
    #                                     t[0]*=-1
    #                                 t=t[:aux.Tpack,:,0,:]
    #                                 t=t[:,momMap]
    #                                 abList=[0,1,4,5] if case=='fw' else [10,11,14,15]
    #                                 t=t[:,:,abList]
    #                                 fw.create_dataset('data/'+src_new+'/'+ky_new,data=t.astype('complex64'))
    #                 except:
    #                     print('flag_cfg_fail: '+cfg+', '+infile)
    #                     1/0
    #     os.remove(outfile_flag)

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
                if not file.endswith('protonup.h5'):
                    continue
                tfStr=file.split('_')[2][2:]
                if int(tfStr) not in aux.tfList:
                    continue
                infileup=inpath+file; infiledn=inpath+file[:-5]+'dn.h5' 
                try:
                    with h5py.File(infileup) as fru, h5py.File(infiledn) as frd:
                        for src in fru.keys():
                            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                            src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                            if flag_setup:
                                moms_old=fru[src]['mvec'][:]*t_flipNJNpc[None,:]
                                moms=opabsDic['post'][aux.set2key({'N','j'})]
                                dic=aux.moms2dic(moms_old)
                                # momMap=[dic[tuple(mom)] if tuple(mom) in dic else -1 for mom in moms]   
                                momMap=[dic[tuple(mom)] for mom in moms]             
                                fw.create_dataset('moms',data=moms)
                                fw.create_dataset('inserts',data=aux.gjList)
                                flag_setup=False

                            tu=fru[src]['MprotonUp'][:]; td=frd[src]['MprotonDn'][:]
                            for j in ['j+','j-']:
                                sgn=1 if j=='j+' else -1
                                t=tu+td*sgn
                                t=t[...,0]+1j*t[...,1]
                                t=t[:,:,:,t_flip_ab]
                                t=np.transpose(t[:,momMap],[0,1,3,2])
                                tfStr=file.split('_')[2][2:]
                                if flags['sign_pf1']:
                                    phase=np.array([get_phase((sx,sy,sz,st),mom[3:6]) for mom in moms])
                                    t=t*phase[None,:,None,None]
                                
                                if flags['convert_gms']:
                                    t_gmMap=np.array([0,1,2,3,4, 5,6,7,8,9, 12,11,10,13,14, 15])
                                    t=t[...,t_gmMap]
                                    t_gmMul=np.array([1,1,1,1,1, 1,1,1,1,1, 1j,-1j,1j,1j,1j, 1j])
                                    t=t*t_gmMul[None,None,None,:]
                                
                                fw.create_dataset('data/'+src_new+'/p_'+j+'_p_deltat_'+tfStr,data=t.astype('complex64'))
                except:
                    print('flag_cfg_fail: '+cfg+', '+infileup+' or '+infiledn)
                    1/0
                    
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()