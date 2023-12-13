'''
nohup python3 -u setupData2_tar.py > log/setupData2_tar.out &
'''


import h5py, os, tarfile, re
import numpy as np


inpath='/p/arch/hch02/hch02b/cD211.54.96/nucl_all_source.tar'
outpath='data_post/'

# pf1 pf2 pc pi1 pi2
Nmax=4
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
base_momList=[[x,y,z] for x in t_range for y in t_range for z in t_range if np.linalg.norm([x,y,z])**2<=Nmax]
base_momList.sort()
target_momList=[mom+[0,0,0]+[0,0,0]+mom+[0,0,0] for mom in base_momList]
target_momList.sort()
target_momList=np.array(target_momList)
target_pf1List=target_momList[:,:3]
target_pf2List=target_momList[:,3:6]
target_pcList=target_momList[:,6:9]
target_pi1List=target_momList[:,9:12]
target_pi2List=target_momList[:,12:15]


case='N.h5_nucl_all_source'


tar = tarfile.open(inpath)
for tarinfo in tar:
    handle = tar.extractfile(tarinfo)
    cfg = tarinfo.name.split('/')[1].split('.')[0]
    with h5py.File(handle) as fr, h5py.File(outpath+cfg+'/'+case, 'w') as fw:
        moms=fr['mvec']
        momDic={}
        for i,mom in enumerate(moms):
            momDic[tuple(mom)]=i
        momMap=np.array([momDic[tuple(mom)] for mom in target_pf1List])

        for src in fr['baryons/nucl_nucl/twop_baryon_1'].keys():
            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
            src_new='sx'+str(sx)+'sy'+str(sy)+'sz'+str(sz)+'st'+str(st)

            for ky in ['twop_baryon_1','twop_baryon_2']:
                ky_new={'twop_baryon_1':'N1,N1','twop_baryon_2':'N2,N2'}[ky]
                tF=fr['baryons/nucl_nucl'][ky][src][:]
                t=tF[...,0]+1j*tF[...,1]
                t=t[:,momMap,:]
                fw.create_dataset('data/'+src_new+'/'+ky_new,data=t)

    print(cfg)
    
print('Done!')