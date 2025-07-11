import h5py,os,re,click
import numpy as np

ens='cB211.072.64'
inpath=f'/p/project/ngff/li47/code/scratch/run/02_discNJN_1D/{ens}/data_avgsrc/'
inpath_loop=f'/p/project/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post/'
outpath=f'/p/project/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_earlytest/test.h5'

tfs=range(3,26+1)
tfs=[6]

with h5py.File(outpath,'w') as fw:
    path='data_aux/cfgs_run'
    with open(path,'r') as f:
        cfgs=f.read().splitlines()
    cfgs.sort()
    
    t1s=[]; t2s=[]
    for i,cfg in enumerate(cfgs):
        print(f'N, {i}/{len(cfgs)}',end='          \r')
        
        path=inpath+cfg+'/N.h5'
        with h5py.File(path) as f:
            # print(f.keys())
            moms=[tuple(mom) for mom in f['moms'][:]]
        
            ind=moms.index((0,0,0))
            t=f['data/N_N'][:,ind]
            t_bw=f['data_bw/N_N'][:,ind]
            t_bw=-np.concatenate([[0],np.flip(t_bw)])
            t=(t+t_bw)/2
            t1s.append(t)
            
            inds=[moms.index(mom) for mom in [(1,0,0),(-1,0,0),(0,-1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
            t=f['data/N_N'][:,:]
            t=np.mean(t[:,inds],axis=1)
            t_bw=f['data_bw/N_N'][:,:]
            t_bw=np.mean(t_bw[:,inds],axis=1)
            t_bw=-np.concatenate([[0],np.flip(t_bw)])
            t=(t+t_bw)/2
            t2s.append(t)
        # break
    fw.create_dataset('N_mom0',data=t1s)
    fw.create_dataset('N_mom1',data=t2s)
    
    # flas=['j+','js','jc']
    # for fla in flas:
    #     t1s={tf:[] for tf in tfs}
    #     t2s={tf:[] for tf in tfs}
    #     t2s_bw={tf:[] for tf in tfs}
    #     tjs=[]
    #     for i,cfg in enumerate(cfgs):
    #         print(f'{fla}, {i}/{len(cfgs)}',end='          \r')
            
    #         path=inpath_loop+cfg+'/j.h5'
    #         with h5py.File(path) as f:
    #             moms=[tuple(mom) for mom in f['moms'][:]]
    #             inserts=[insert.decode() for insert in f['inserts;g{m,Dn};tl']]
                
    #             i_mom=moms.index((0,0,0))
    #             i_insert=inserts.index('tt')
                
    #             t=np.mean(f['data'][fla+';g{m,Dn};tl'][:,i_mom,i_insert],axis=0)
    #             tjs.append(t)
            
    #         path=inpath+cfg+'/discNJN_'+fla+';g{m,Dn};tl.h5'
    #         with h5py.File(path) as f:
    #             # print(f['notes'][:])
    #             # print(f.keys())
                
    #             moms=[tuple(mom) for mom in f['moms'][:]]
    #             i_mom=moms.index((0,0,0,0,0,0))    
                
    #             inserts=[insert.decode() for insert in f['inserts;g{m,Dn};tl']]
    #             i_insert=inserts.index('tt')
                
    #             for tf in tfs:
    #                 t=f['data']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
    #                 t_bw=-f['data_bw']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
    #                 t=(t+t_bw)/2
    #                 t1s[tf].append(t)
                
    #             cases=[
    #                 [(1,0,0,0,0,0),'tx',1],[(-1,0,0,0,0,0),'tx',-1],
    #                 [(0,1,0,0,0,0),'ty',1],[(0,-1,0,0,0,0),'ty',-1],
    #                 [(0,0,1,0,0,0),'tz',1],[(0,0,-1,0,0,0),'tz',-1],
    #             ]
    #             for tf in tfs:
    #                 t=0; t_bw=0
    #                 for mom,insert,factor in cases:
    #                     i_mom=moms.index(mom); i_insert=inserts.index(insert)
    #                     t += factor * f['data']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
    #                     t_bw += factor * f['data_bw']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
    #                 t2s[tf].append(t)
    #                 t2s_bw[tf].append(t_bw)
    #         # break
    #     for tf in tfs:
    #         fw.create_dataset(f'{fla}/P44(G0,0,0)/{tf}',data=t1s[tf])
    #         fw.create_dataset(f'{fla}/P4i(G0,pi,pi)/{tf}',data=t2s[tf])
    #         fw.create_dataset(f'{fla}/P4i(G0,pi,pi)_bw/{tf}',data=t2s_bw[tf])
    #     fw.create_dataset(f'{fla}/P44(G0,0,0)_vev',data=tjs)
    #     # break
    
    cases=[
        [(1,0,0,0,0,0),'tx',1],[(-1,0,0,0,0,0),'tx',-1],
        [(0,1,0,0,0,0),'ty',1],[(0,-1,0,0,0,0),'ty',-1],
        [(0,0,1,0,0,0),'tz',1],[(0,0,-1,0,0,0),'tz',-1],
    ]

    for mom,insert,factor in cases:
        name=insert+('p' if factor==1 else 'n')
        flas=['j+']
        for fla in flas:
            t1s={tf:[] for tf in tfs}
            t2s={tf:[] for tf in tfs}
            t2s_bw={tf:[] for tf in tfs}
            tjs=[]
            for i,cfg in enumerate(cfgs):
                print(f'{name}, {fla}, {i}/{len(cfgs)}',end='          \r')
                
                path=inpath+cfg+'/discNJN_'+fla+';g{m,Dn};tl.h5'
                with h5py.File(path) as f:
                    moms=[tuple(mom) for mom in f['moms'][:]]
                    inserts=[insert.decode() for insert in f['inserts;g{m,Dn};tl']]
                    

                    for tf in tfs:
                        t=0; t_bw=0
                        i_mom=moms.index(mom); i_insert=inserts.index(insert)
                        t += factor * f['data']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
                        t_bw += factor * f['data_bw']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
                        t2s[tf].append(t)
                        t2s_bw[tf].append(t_bw)
                # break
            for tf in tfs:
                fw.create_dataset(f'{fla}/P4i(G0,pi,pi)/{name}/{tf}',data=t2s[tf])
                fw.create_dataset(f'{fla}/P4i(G0,pi,pi)_bw/{name}/{tf}',data=t2s_bw[tf])
            # break
