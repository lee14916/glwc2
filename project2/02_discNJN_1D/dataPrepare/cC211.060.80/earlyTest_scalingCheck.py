'''
nohup python3 -u earlyTest_scalingCheck.py > log/earlyTest_scalingCheck.out &
'''
import h5py,os,re,click
import numpy as np

ens='cC211.060.80'
inpath=f'/p/project/ngff/li47/code/scratch/run/02_discNJN_1D/{ens}/data_avgsrc_scalingCheck/'
outpath=f'/p/project/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_earlytest/test_scalingCheck.h5'

tfs={'cB211.072.64':range(2,26+1),'cC211.060.80':range(2,20+1,2),'cD211.054.96':range(2,32+1)}[ens]

with h5py.File(outpath,'w') as fw:
    path='data_aux/cfgs_run'
    with open(path,'r') as f:
        cfgs=f.read().splitlines()
    cfgs.sort()
    
    t1s=[]; t2s=[]
    t3s={mom:[] for mom in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]}
    for i,cfg in enumerate(cfgs):
        print(f'N, {i}/{len(cfgs)}',end='          \r')
        
        path=inpath+cfg+'/discNJN_js;g{m,Dn};tl.h5'
        with h5py.File(path) as f:
            # print(f.keys())
            moms=[tuple(mom) for mom in f['moms'][:,:3]]
        
            ind=moms.index((0,0,0))
            t=f['dataN'][:,:,ind,0]
            t_bw=f['dataN_bw'][:,:,ind,0]
            t_bw=-np.concatenate([np.zeros([len(t_bw),1]),np.flip(t_bw,axis=1)],axis=1)
            t=(t+t_bw)/2
            t1s.append(t)
            
            inds=[moms.index(mom) for mom in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
            t=f['dataN'][:,:,:,0]
            t=np.mean(t[:,:,inds],axis=2)
            t_bw=f['dataN_bw'][:,:,:,0]
            t_bw=np.mean(t_bw[:,:,inds],axis=2)
            t_bw=-np.concatenate([np.zeros([len(t_bw),1]),np.flip(t_bw,axis=1)],axis=1)
            t=(t+t_bw)/2
            t2s.append(t)
            
            # for mom in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            #     ind=moms.index(mom)
            #     t=f['data/N_N'][:,ind]
            #     t3s[mom].append(t)
        # break
    fw.create_dataset('N_mom0',data=t1s)
    fw.create_dataset('N_mom1',data=t2s)
    # for mom in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
    #     fw.create_dataset(f'N_mom_{mom}',data=t3s[mom])
    
    for t_data in ['data1','data2','data3','data4']:
        flas=['j+','js','jc'][1:2]
        for fla in flas:
            t1s={tf:[] for tf in tfs}
            t2s={tf:[] for tf in tfs}
            t2s_bw={tf:[] for tf in tfs}
            for i,cfg in enumerate(cfgs):
                print(f'{fla}, {i}/{len(cfgs)}',end='          \r')
                
                
                path=inpath+cfg+'/discNJN_'+fla+';g{m,Dn};tl.h5'
                with h5py.File(path) as f:
                    
                    moms=[tuple(mom) for mom in f['moms'][:]]
                    i_mom=moms.index((0,0,0,0,0,0))    
                    
                    inserts=[insert.decode() for insert in f['inserts;g{m,Dn};tl']]
                    i_insert=inserts.index('tt')
                    
                    # for tf in tfs:
                    #     t=f['data']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,:,i_mom,0,i_insert]
                    #     t_bw=-f['data_bw']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,:,i_mom,0,i_insert]
                    #     t=(t+t_bw)/2
                    #     t1s[tf].append(t)
                    
                    cases=[
                        [(1,0,0,0,0,0),'tx',1],[(-1,0,0,0,0,0),'tx',-1],
                        [(0,1,0,0,0,0),'ty',1],[(0,-1,0,0,0,0),'ty',-1],
                        [(0,0,1,0,0,0),'tz',1],[(0,0,-1,0,0,0),'tz',-1],
                    ]
                    for tf in tfs:
                        t=0; t_bw=0
                        for mom,insert,factor in cases:
                            i_mom=moms.index(mom); i_insert=inserts.index(insert)
                            t += factor * f[t_data]['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,:,i_mom,0,i_insert]
                            t_bw += factor * f[f'{t_data}_bw']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,:,i_mom,0,i_insert]
                        t=(t+t_bw)/2/6
                        t2s[tf].append(t)
                # break
            for tf in tfs:
                # fw.create_dataset(f'{fla}/P44(G0,0,0)/{tf}',data=t1s[tf])
                fw.create_dataset(f'{t_data}/{fla}/P4i(G0,pi,pi)/{tf}',data=t2s[tf])