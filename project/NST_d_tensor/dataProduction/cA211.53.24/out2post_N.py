
'''
cat data_aux/cfgs_run | xargs -n 1 -I @ -P 10 python3 -u out2post_N.py -c @ > log/out2post_N.out & 
'''
import os, click, h5py, re, pickle
import numpy as np
import aux

1/0

# postcode='N-a-Nsrc4*32'
# def cfg2out(cfg):
#     path='/p/scratch/pines/fpittler/run/nucleon_sigma_term/cA2.09.48/NN/outputdata_sinkMom/'+cfg[1:]+'/'   
#     return path

flags={
    'BWZ_globalPhase':True, # -1j
    # following will not be used any more.
    'D1ii_pi2Phase':False, # e^{-i*pi2*x}
    'D1ii_normalize':False, # 1/12
    'T_seq2526':False, # -1
}

t_D1ii_normalize=1/12 if flags['D1ii_normalize'] else 1
t_T_seq2526=-1 if flags['T_seq2526'] else 1
t_BWZ_globalPhase=-1j if flags['BWZ_globalPhase'] else 1


tu=1/np.sqrt(2) # pi0 u
td=-1/np.sqrt(2) # pi0 d

sumDic={}
# sumDic['N_2pt']={
#     'p_p':[(1,'NP')],
#     'n_n':[(1,'N0')]
# }

t_factor=t_T_seq2526
sumDic['T']={
    'p_n,pi+':[(1,'Tseq'+str(i)) for i in [11,12,13,14]],
    'p_p,pi0':[(tu,'Tseq'+str(i)) for i in [21,22,23,24]]+[(td*t_factor,'Tseq'+str(i)) for i in [25,26]],
}

# t_factor=t_D1ii_normalize
# sumDic['D1ii']={
#     'n,pi+_p':[(1*t_factor,'D1ii'+str(i)) for i in [13,14,15,16]],
#     'p,pi0_p':[(tu*t_factor,'D1ii'+str(i)) for i in [1,2,3,4]]+[(td*t_factor,'D1ii'+str(i)) for i in [9,10]],
# }

# sumDic['B_2pt']={
#     'n,pi+_n,pi+':[(1,'B'+str(i)) for i in [13,14,15,16]],
#     'n,pi+_p,pi0':[(tu,'B'+str(i)) for i in [17,18,19,20]]+[(td,'B'+str(i)) for i in []],
#     'p,pi0_n,pi+':[(tu,'B'+str(i)) for i in [9,10,11,12]]+[(td,'B'+str(i)) for i in []],
#     'p,pi0_p,pi0':[(tu*tu,'B'+str(i)) for i in [3,4,5,6]]+[(tu*td,'B'+str(i)) for i in []]+\
#         [(td*tu,'B'+str(i)) for i in []]+[(td*td,'B'+str(i)) for i in [7,8]],
# }
# sumDic['W_2pt']={
#     'n,pi+_n,pi+':[(1,'W'+str(i)) for i in [25,26,27,28]],
#     'n,pi+_p,pi0':[(tu,'W'+str(i)) for i in [29,30,31,32]]+[(td,'W'+str(i)) for i in [33,34,35,36]],
#     'p,pi0_n,pi+':[(tu,'W'+str(i)) for i in [17,18,19,20]]+[(td,'W'+str(i)) for i in [21,22,23,24]],
#     'p,pi0_p,pi0':[(tu*tu,'W'+str(i)) for i in [5,6,7,8]]+[(tu*td,'W'+str(i)) for i in [9,10,11,12]]+\
#         [(td*tu,'W'+str(i)) for i in [13,14,15,16]]+[(td*td,'W'+str(i)) for i in []],
# }
# sumDic['Z_2pt']={
#     'n,pi+_n,pi+':[(1,'Z'+str(i)) for i in [15,16]],
#     'n,pi+_p,pi0':[(tu,'Z'+str(i)) for i in []]+[(td,'Z'+str(i)) for i in [17,18,19,20]],
#     'p,pi0_n,pi+':[(tu,'Z'+str(i)) for i in []]+[(td,'Z'+str(i)) for i in [11,12,13,14]],
#     'p,pi0_p,pi0':[(tu*tu,'Z'+str(i)) for i in [5,6,7,8]]+[(tu*td,'Z'+str(i)) for i in []]+\
#         [(td*tu,'Z'+str(i)) for i in []]+[(td*td,'Z'+str(i)) for i in [9,10]],
# }
# sumDic['M_correct_2pt']={
#     'p,pi+_p,pi+':[(1,'MNPPP')],
#     'n,pi+_n,pi+':[(1,'MN0PP')],
#     'p,pi0_p,pi0':[(tu*tu,'MNPP01')]+[(td*td,'MNPP02')],
# }

# t_factor=t_BWZ_globalPhase
# sumDic['B']={
#     'p_j+_n,pi+':[(1*t_factor,'B'+str(i)) for i in [9,10,11,12]]+[(1*t_factor,'B'+str(i)) for i in []],
#     'p_j-_n,pi+':[(1*t_factor,'B'+str(i)) for i in [9,10,11,12]]+[(-1*t_factor,'B'+str(i)) for i in []],

#     'p_j+_p,pi0':[(tu*t_factor,'B'+str(i)) for i in [3,4,5,6]]+[(td*t_factor,'B'+str(i)) for i in []]+\
#         [(tu*t_factor,'B'+str(i)) for i in []]+[(td*t_factor,'B'+str(i)) for i in [7,8]],
#     'p_j-_p,pi0':[(tu*t_factor,'B'+str(i)) for i in [3,4,5,6]]+[(td*t_factor,'B'+str(i)) for i in []]+\
#         [(-tu*t_factor,'B'+str(i)) for i in []]+[(-td*t_factor,'B'+str(i)) for i in [7,8]],
# }
# sumDic['W']={
#     'p_j+_n,pi+':[(1*t_factor,'W'+str(i)) for i in [17,18,19,20]]+[(1*t_factor,'W'+str(i)) for i in [21,22,23,24]],
#     'p_j-_n,pi+':[(1*t_factor,'W'+str(i)) for i in [17,18,19,20]]+[(-1*t_factor,'W'+str(i)) for i in [21,22,23,24]],

#     'p_j+_p,pi0':[(tu*t_factor,'W'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'W'+str(i)) for i in [9,10,11,12]]+\
#         [(tu*t_factor,'W'+str(i)) for i in [13,14,15,16]]+[(td*t_factor,'W'+str(i)) for i in []],
#     'p_j-_p,pi0':[(tu*t_factor,'W'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'W'+str(i)) for i in [9,10,11,12]]+\
#         [(-tu*t_factor,'W'+str(i)) for i in [13,14,15,16]]+[(-td*t_factor,'W'+str(i)) for i in []],
# }
# sumDic['Z']={
#     'p_j+_n,pi+':[(1*t_factor,'Z'+str(i)) for i in []]+[(1*t_factor,'Z'+str(i)) for i in [11,12,13,14]],
#     'p_j-_n,pi+':[(1*t_factor,'Z'+str(i)) for i in []]+[(-1*t_factor,'Z'+str(i)) for i in [11,12,13,14]],

#     'p_j+_p,pi0':[(tu*t_factor,'Z'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'Z'+str(i)) for i in []]+\
#         [(tu*t_factor,'Z'+str(i)) for i in []]+[(td*t_factor,'Z'+str(i)) for i in [9,10]],
#     'p_j-_p,pi0':[(tu*t_factor,'Z'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'Z'+str(i)) for i in []]+\
#         [(-tu*t_factor,'Z'+str(i)) for i in []]+[(-td*t_factor,'Z'+str(i)) for i in [9,10]],
# }


@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=cfg2out(cfg)
    outpath='data_post/'+cfg+'/'
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
            flag_op=False
            with h5py.File(t_outfile,'w') as fw:
                for file in files:
                    if not file.endswith('N.h5'):
                        continue
                    infile=inpath+file
                    with h5py.File(infile) as fr:
                        for src in fr.keys():
                            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                            src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                            
                            if not flag_op:
                                moms_old=fr[src]['mvec'][:]
                                moms=opabsDic['post'][aux.set2key({'N'})]
                                dic=aux.moms2dic(moms_old)
                                mom_flip = -1 if case == 'bw' else 1
                                momMap=[dic[tuple(mom[3:6])] for mom in mom_flip*np.array(moms)]                           
                                fw.create_dataset('moms',data=moms)
                                flag_op=True
                            
                            ky2new={'NP':'p_p','N0':'n_n'}
                            for ky,ky_new in ky2new.items():
                                t=fr[src][ky][:]
                                t=t[...,0]+1j*t[...,1]
                                if case=='bw':
                                    t=-np.roll(np.flip(t,axis=0),1,axis=0)
                                    t[0]*=-1
                                t=t[:aux.Tpack,:,0,:]
                                t=t[:,momMap]
                                abList=[0,1,4,5] if case=='fw' else [10,11,14,15]
                                t=t[:,:,abList]
                                fw.create_dataset('data/'+src_new+'/'+ky_new,data=t.astype('complex64'))
        os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)

run()