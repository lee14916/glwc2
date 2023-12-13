'''
nohup python3 -u setupData2.py > log/setupData2.out &
'''

import re
import h5py, os, tarfile
import numpy as np

inpath='data_pre/'
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

# main
cfgs = [cfg for cfg in os.listdir(inpath)]
cfgs.sort()

for cfg in cfgs:
    os.makedirs(outpath+cfg,exist_ok=True)

    # case="N.h5_twop_threep_dt20_64srcs"
    # with h5py.File(outpath+cfg+'/'+case, 'w') as fw:
    #     fw.create_dataset('mvec',data=target_momList)
    #     for file in os.listdir(inpath+cfg):
    #         if file != case:
    #             continue
    #         with h5py.File(inpath+cfg+'/'+file) as fr:
    #             moms=fr['mvec']
    #             momDic={}
    #             for i,mom in enumerate(moms):
    #                 momDic[tuple(mom)]=i
    #             momMap=np.array([momDic[tuple(mom)] for mom in target_pf1List])

    #             for src in fr['baryons/nucl_nucl/twop_baryon_1'].keys():
    #                 (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
    #                 (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
    #                 src_new='sx'+str(sx)+'sy'+str(sy)+'sz'+str(sz)+'st'+str(st)

    #                 for ky in ['twop_baryon_1','twop_baryon_2']:
    #                     ky_new={'twop_baryon_1':'N1,N1','twop_baryon_2':'N2,N2'}[ky]
    #                     tF=fr['baryons/nucl_nucl'][ky][src]
    #                     t=tF[...,0]+1j*tF[...,1]
    #                     t=t[:,momMap,:]
    #                     fw.create_dataset('data/'+src_new+'/'+ky_new,data=t)

    # case="N.h5_twop_threep_2"
    # with h5py.File(outpath+cfg+'/'+case, 'w') as fw:
    #     fw.create_dataset('mvec',data=target_momList)
    #     for file in os.listdir(inpath+cfg):
    #         if file != case:
    #             continue
    #         with h5py.File(inpath+cfg+'/'+file) as fr:
    #             moms=fr['mvec']
    #             momDic={}
    #             for i,mom in enumerate(moms):
    #                 momDic[tuple(mom)]=i
    #             momMap=np.array([momDic[tuple(mom)] for mom in target_pf1List])

    #             for src in fr['baryons/nucl_nucl/twop_baryon_1'].keys():
    #                 (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
    #                 (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
    #                 src_new='sx'+str(sx)+'sy'+str(sy)+'sz'+str(sz)+'st'+str(st)

    #                 for ky in ['twop_baryon_1','twop_baryon_2']:
    #                     ky_new={'twop_baryon_1':'N1,N1','twop_baryon_2':'N2,N2'}[ky]
    #                     tF=fr['baryons/nucl_nucl'][ky][src]
    #                     t=tF[...,0]+1j*tF[...,1]
    #                     t=t[:,momMap,:]
    #                     fw.create_dataset('data/'+src_new+'/'+ky_new,data=t)

    case="N.h5_hch02k_twop"
    with h5py.File(outpath+cfg+'/'+case, 'w') as fw:
        fw.create_dataset('mvec',data=target_momList)
        for file in os.listdir(inpath+cfg):
            if file != case:
                continue
            
            infile=inpath+cfg+'/'+file
            tar = tarfile.open(infile)
            for tarinfo in tar:
                if 'baryons' not in tarinfo.name:
                    continue
                handle = tar.extractfile(tarinfo)
                with h5py.File(handle) as fr:
                    moms=fr['Momenta_list_xyz']
                    momDic={}
                    for i,mom in enumerate(moms):
                        momDic[tuple(mom)]=i
                    momMap=np.array([momDic[tuple(mom)] for mom in target_pf1List])
                    ky_cfg='conf_'+cfg[:4]
                    for src in fr[ky_cfg].keys():
                        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                        src_new='sx'+str(sx)+'sy'+str(sy)+'sz'+str(sz)+'st'+str(st)

                        for ky in ['twop_baryon_1','twop_baryon_2']:
                            ky_new={'twop_baryon_1':'N1,N1','twop_baryon_2':'N2,N2'}[ky]
                            t=fr[ky_cfg][src]['nucl_nucl'][ky][:]
                            t=t[...,0]+1j*t[...,1]
                            t=t[:,momMap,:]
                            fw.create_dataset('data/'+src_new+'/'+ky_new,data=t)

    print(cfg)
    # print()
    # break
    
print('Done!')