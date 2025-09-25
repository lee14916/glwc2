'''
nohup python3 -u earlyTest.py > log/earlyTest.out &
'''
import h5py,os,re,click
import numpy as np

ens='cD211.054.96'
inpath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run1/{ens}/data_avgsrc/'
inpath_loop=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post/'
outpath=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_earlytest/test2.h5'

# tfs={'cB211.072.64':range(2,26+1),'cC211.060.80':range(2,28+1),'cD211.054.96':range(2,32+1),'cE211.044.112':range(2,32+1)}[ens]
tfs={'cB211.072.64':range(2,22+1),'cC211.060.80':range(2,26+1),'cD211.054.96':range(2,30+1),'cE211.044.112':range(2,32+1)}[ens]

stouts=range(40+1)

with h5py.File(outpath,'w') as fw:
    path='data_aux/cfgs_run'
    with open(path,'r') as f:
        cfgs=f.read().splitlines()
    cfgs.sort()
    
    t1s=[]; t2s=[]
    t3s={mom:[] for mom in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]}
    for i,cfg in enumerate(cfgs):
        print(f'N, {i}/{len(cfgs)}',end='          \r')
        
        path=inpath+cfg+'/N.h5'
        with h5py.File(path) as f:
            # print(f.keys())
            moms=[tuple(mom) for mom in f['moms'][:]]
        
            ind=moms.index((0,0,0))
            t=f['data/N_N'][:,ind]
            t1s.append(t)
            
            inds=[moms.index(mom) for mom in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
            t=f['data/N_N'][:,:]
            t=np.mean(t[:,inds],axis=1)
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
    
    flas=['j+','j-','js','jc']
    for fla in flas:
        t1s={tf:[] for tf in tfs}
        t2s={tf:[] for tf in tfs}
        tjs=[]
        for i,cfg in enumerate(cfgs):
            print(f'{fla}, {i}/{len(cfgs)}',end='          \r')
            
            path=inpath_loop+cfg+'/j.h5'
            with h5py.File(path) as f:
                moms=[tuple(mom) for mom in f['moms'][:]]
                inserts=[insert.decode() for insert in f['inserts;g{m,Dn};tl']]
                
                i_mom=moms.index((0,0,0))
                i_insert=inserts.index('tt')
                
                t=np.mean(f['data'][fla+';g{m,Dn};tl'][:,i_mom,i_insert],axis=0)
                tjs.append(t)
            
            path=inpath+cfg+'/discNJN_'+fla+';g{m,Dn};tl.h5'
            with h5py.File(path) as f:
                # print(f['notes'][:])
                # print(f.keys())
                
                moms=[tuple(mom) for mom in f['moms'][:]]
                i_mom=moms.index((0,0,0,0,0,0))    
                
                inserts=[insert.decode() for insert in f['inserts;g{m,Dn};tl']]
                i_insert=inserts.index('tt')
                
                for tf in tfs:
                    t=f['data']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
                    t1s[tf].append(t)
                
                cases=[
                    [(1,0,0,0,0,0),'tx',1],[(-1,0,0,0,0,0),'tx',-1],
                    [(0,1,0,0,0,0),'ty',1],[(0,-1,0,0,0,0),'ty',-1],
                    [(0,0,1,0,0,0),'tz',1],[(0,0,-1,0,0,0),'tz',-1],
                ]
                for tf in tfs:
                    t=0
                    for mom,insert,factor in cases:
                        i_mom=moms.index(mom); i_insert=inserts.index(insert)
                        t += factor * f['data']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
                    t=t/6
                    t2s[tf].append(t)
            # break
        for tf in tfs:
            fw.create_dataset(f'{fla}/P44(G0,0,0)/{tf}',data=t1s[tf])
            fw.create_dataset(f'{fla}/P4i(G0,pi,pi)/{tf}',data=t2s[tf])
        fw.create_dataset(f'{fla}/P44(G0,0,0)_vev',data=tjs)
        # break
        
    flas=[f'jg;stout{stout}' for stout in stouts]
    for fla in flas:
        t1s={tf:[] for tf in tfs}
        t2s={tf:[] for tf in tfs}
        tjs=[]
        for i,cfg in enumerate(cfgs):
            print(f'{fla}, {i}/{len(cfgs)}',end='          \r')
            
            path=inpath_loop+cfg+'/j.h5'
            with h5py.File(path) as f:
                moms=[tuple(mom) for mom in f['moms'][:]]
                inserts=[insert.decode() for insert in f['inserts;jg']]
                
                i_mom=moms.index((0,0,0))
                i_insert=inserts.index('tt')
                
                t=np.mean(f['data'][fla][:,i_mom,i_insert],axis=0)
                tjs.append(t)
            
            path=inpath+cfg+'/discNJN_jg;stout.h5'
            with h5py.File(path) as f:
                # print(f['notes'][:])
                # print(f.keys())
                
                moms=[tuple(mom) for mom in f['moms'][:]]
                i_mom=moms.index((0,0,0,0,0,0))    
                
                inserts=[insert.decode() for insert in f['inserts;jg']]
                i_insert=inserts.index('tt')
                
                for tf in tfs:
                    t=f['data']['N_N_'+fla+'_'+str(tf)][:,i_mom,0,i_insert]
                    t1s[tf].append(t)
                
                cases=[
                    [(1,0,0,0,0,0),'tx',1],[(-1,0,0,0,0,0),'tx',-1],
                    [(0,1,0,0,0,0),'ty',1],[(0,-1,0,0,0,0),'ty',-1],
                    [(0,0,1,0,0,0),'tz',1],[(0,0,-1,0,0,0),'tz',-1],
                ]
                for tf in tfs:
                    t=0
                    for mom,insert,factor in cases:
                        i_mom=moms.index(mom); i_insert=inserts.index(insert)
                        t += factor * f['data']['N_N_'+fla+'_'+str(tf)][:,i_mom,0,i_insert]
                    t=t/6
                    t2s[tf].append(t)
            # break
        for tf in tfs:
            fw.create_dataset(f'{fla}/P44(G0,0,0)/{tf}',data=t1s[tf])
            fw.create_dataset(f'{fla}/P4i(G0,pi,pi)/{tf}',data=t2s[tf])
        fw.create_dataset(f'{fla}/P44(G0,0,0)_vev',data=tjs)
        # break

    
    # cases=[
    #     [(1,0,0,0,0,0),'tx',1],[(-1,0,0,0,0,0),'tx',-1],
    #     [(0,1,0,0,0,0),'ty',1],[(0,-1,0,0,0,0),'ty',-1],
    #     [(0,0,1,0,0,0),'tz',1],[(0,0,-1,0,0,0),'tz',-1],
    # ]

    # for mom,insert,factor in cases:
    #     name=insert+('p' if factor==1 else 'n')
    #     flas=['j+']
    #     for fla in flas:
    #         t1s={tf:[] for tf in tfs}
    #         t2s={tf:[] for tf in tfs}
    #         t2s_bw={tf:[] for tf in tfs}
    #         tjs=[]
    #         for i,cfg in enumerate(cfgs):
    #             print(f'{name}, {fla}, {i}/{len(cfgs)}',end='          \r')
                
    #             path=inpath+cfg+'/discNJN_'+fla+';g{m,Dn};tl.h5'
    #             with h5py.File(path) as f:
    #                 moms=[tuple(mom) for mom in f['moms'][:]]
    #                 inserts=[insert.decode() for insert in f['inserts;g{m,Dn};tl']]
                    

    #                 for tf in tfs:
    #                     t=0; t_bw=0
    #                     i_mom=moms.index(mom); i_insert=inserts.index(insert)
    #                     t += factor * f['data']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
    #                     t_bw += factor * f['data_bw']['N_N_'+fla+';g{m,Dn};tl_'+str(tf)][:,i_mom,0,i_insert]
    #                     t2s[tf].append(t)
    #                     t2s_bw[tf].append(t_bw)
    #             # break
    #         for tf in tfs:
    #             fw.create_dataset(f'{fla}/P4i(G0,pi,pi)/{name}/{tf}',data=t2s[tf])
    #             fw.create_dataset(f'{fla}/P4i(G0,pi,pi)_bw/{name}/{tf}',data=t2s_bw[tf])
    #         # break
