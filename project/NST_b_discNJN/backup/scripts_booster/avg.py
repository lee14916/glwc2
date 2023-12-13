'''
nohup python3 -u avg.py > log/avg.out &
'''

import os, h5py, re
import numpy as np

ensembles=['cA2.09.48']
ensembles=['cD211.054.96']
# ensembles=['cA2.09.48','cB211.072.64','cC211.060.80','cD211.054.96']
# ensembles=['cB211.072.64']
for ens in ensembles:
    print(ens)
    path_cfgs=ens+'/cfgs_final'
    inpath=ens+'/data_post/'
    outpath=ens+'/data_avg/'
    outpath_merge=ens+'/data_merge/'

    tfList=[10,12,14]
    tfList=[6,8,10,12,14,16,18,20,22,24,26]

    if ens=='cA2.09.48':
        (lat_L,lat_T)=(48,96)
    elif ens=='cB211.072.64':
        (lat_L,lat_T)=(64,128)
    elif ens=='cC211.060.80':
        (lat_L,lat_T)=(80,160)
    elif ens=='cD211.054.96':
        (lat_L,lat_T)=(96,192)


    # pf1 pf2 pc pi1 pi2; pf1+pf2+pc=pi1+pi2
    Nmax=1
    Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
    base_momList=[[x,y,z] for x in t_range for y in t_range for z in t_range if np.linalg.norm([x,y,z])**2<=Nmax]
    base_momList.sort()
    target_momLists={}
    target_momLists['N']=[mom+[0,0,0]+[0,0,0]+mom+[0,0,0] for mom in base_momList]
    target_momLists['NJN']=[mom1+[0,0,0]+mom2+list(np.array(mom1)+mom2)+[0,0,0] for mom1 in base_momList for mom2 in base_momList]
    for ky in target_momLists:
        t=target_momLists[ky]
        t=[mom for mom in t if np.linalg.norm(mom[9:12])**2<=Nmax]
        t.sort()
        target_momLists[ky]=np.array(t)

    def get_st(src):
        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
        return int(st)

    def get_phase(src,mom):
        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
        return np.exp(1j*(2*np.pi/lat_L)*(np.array([sx,sy,sz])@mom))

    #main
    with open(path_cfgs) as f:
        cfgs=f.read().splitlines()
    cfgs.sort()
    
    # print(cfgs)

    for cfg in cfgs:
        os.makedirs(outpath+cfg,exist_ok=True)
    
        srcsAll=[]; tr={}
        kys=['N1,N1','N2,N2']
        for ky in kys:
            tr[ky]=[]
        
        for file in os.listdir(inpath+cfg):
            if not file.startswith('N.h5'):
                continue
            # print(file)
            with h5py.File(inpath+cfg+'/'+file) as fr:
                moms=fr['mvec']
                momDic={}
                for i,mom in enumerate(moms):
                    momDic[tuple(mom)]=i

                srcs = list(fr['data'].keys())
                srcsAll += srcs
                for ky in kys:
                    tr[ky] += [fr['data'][src][ky][:] for src in srcs]
        for ky in kys:
            tr[ky]=np.array(tr[ky])
        
        with h5py.File(inpath+cfg+'/j.h5') as fj:
            moms=fj['mvec']
            momDic_j={}
            for i,mom in enumerate(moms):
                momDic_j[tuple(mom)]=i

            def get_momConserved(mom):
                [pf1,pf2,pc,pi1,pi2]=[list(mom)[i:i+3] for i in range(0,15,3)]
                pi1=list(np.array(pf1)+pf2+pc-pi2)
                return pf1+pf2+pc+pi1+pi2

            with h5py.File(outpath+cfg+'/N.h5','w') as fw:
                fw.create_dataset('mvec',data=target_momLists['N'])
                
                momMap=[momDic[tuple(mom)] for mom in target_momLists['N']]
                for ky in kys:
                    t=np.mean(tr[ky],axis=0)
                    fw.create_dataset('data/'+ky,data=t[:,momMap])

            with h5py.File(outpath+cfg+'/N-j.h5','w') as fw:
                fw.create_dataset('mvec',data=target_momLists['NJN'])
                momMap=[momDic[tuple(get_momConserved(mom*\
                    [1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0]))] for mom in target_momLists['NJN']]
                momMap_j=[momDic_j[tuple(mom*\
                    [0,0,0, 0,0,0, 1,1,1, 0,0,0, 0,0,0])] for mom in target_momLists['NJN']]

                stList=[get_st(src) for src in srcsAll]
                t_phase=np.array([[get_phase(src,mom[6:9]) for mom in fj['mvec']] for src in srcsAll])
                
                for ky in kys:
                    for tf in tfList:
                        timeMap=[tf]*(tf+1)
                        t_base=tr[ky][:,timeMap][:,:,momMap]
                        for kyj in ['j+','j-']:
                            t_j=np.array([np.roll(fj['data'][kyj],-st,axis=0)[:tf+1] for st in stList])
                            t_j=t_j*t_phase[:,None,:,None]
                            t_j=t_j[:,:,momMap_j,:]
                            t_res=np.einsum('stmd,stmg->tmgd',t_base,t_j)/len(t_base)

                            t=ky.split(',')
                            ky_new=','.join([t[0],kyj,t[1]])+'_deltat_'+str(tf)

                            fw.create_dataset('data/'+ky_new,data=t_res)

            with h5py.File(outpath+cfg+'/N-jbw.h5','w') as fw:
                fw.create_dataset('mvec',data=target_momLists['NJN'])
                momMap=[momDic[tuple(get_momConserved(mom*\
                    [1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0]))] for mom in target_momLists['NJN']]
                momMap_j=[momDic_j[tuple(mom*\
                    [0,0,0, 0,0,0, 1,1,1, 0,0,0, 0,0,0])] for mom in target_momLists['NJN']]

                stList=[get_st(src) for src in srcsAll]
                t_phase=np.array([[get_phase(src,mom[6:9]) for mom in fj['mvec']] for src in srcsAll])
                
                for ky in kys:
                    for tf in tfList:
                        timeMap=[-tf]*(tf+1)
                        t_base=tr[ky][:,timeMap][:,:,momMap]
                        for kyj in ['j+','j-']:
                            t_j=np.array([np.roll(fj['data'][kyj],-st-1,axis=0)[::-1][:tf+1] for st in stList])
                            t_j=t_j*t_phase[:,None,:,None]
                            t_j=t_j[:,:,momMap_j,:]
                            t_res=np.einsum('stmd,stmg->tmgd',t_base,t_j)/len(t_base)

                            t=ky.split(',')
                            ky_new=','.join([t[0],kyj,t[1]])+'_deltat_'+str(tf)

                            fw.create_dataset('data/'+ky_new,data=t_res)

        print(cfg)
        # break

    os.makedirs(outpath_merge,exist_ok=True)
    for diag in ['N','N-j','N-jbw']:
    # for diag in ['N']:
        with h5py.File(outpath_merge+diag+'.h5', 'w') as fw:
            mvec_flag=False
            for cfg in cfgs:
                with h5py.File(outpath+cfg+'/'+diag+'.h5') as fr:
                    if not mvec_flag:
                        fr.copy(fr['mvec'],fw,name='mvec')
                        mvec_flag=True
                    fr.copy(fr['data'],fw,name='data/'+cfg)

print('Done!')