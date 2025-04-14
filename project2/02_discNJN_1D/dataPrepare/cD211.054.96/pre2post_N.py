'''
cat data_aux/cfgs_all_N | xargs -I @ -P 10 python3 -u pre2post_N.py -c @ > log/pre2post_N.out & 
'''

import re, click
import h5py, os, tarfile
import numpy as np

ens='cD211.054.96'

basePath=f'/p/project/ngff/li47/code/projectData/02_discNJN_1D/{ens}/'
tmax={'cB211.072.64':36,'cC211.060.80':40,'cD211.054.96':48}[ens]


cfg2old=lambda cfg: cfg[1:]+'_r'+{'a':'0','b':'1','c':'2','d':'3'}[cfg[0]]
cfg2new=lambda cfg: {'0':'a','1':'b','2':'c','3':'d'}[cfg[-1]] + cfg[:4]

Nmax=4
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
moms_N=[[x,y,z] for x in t_range for y in t_range for z in t_range if np.linalg.norm([x,y,z])**2<=Nmax]
moms_N.sort()

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    cfg_old=cfg2old(cfg)
    inpath=f'{basePath}data_pre_N/{cfg_old}/'
    inpath2='/p/arch1/hch02/hch02b/cD211.54.96/nucl_all_source.tar'
    outpath=f'{basePath}data_post/{cfg}/'
    outpathFullmom=f'{basePath}data_N_fullmom/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    os.makedirs(outpathFullmom,exist_ok=True)
    
    files=os.listdir(inpath)
    
    file="N.h5_twop_threep_3"
    if file in files:
        outfile=f'{outpath}{file}'
        outfile_flag=outfile+'_flag'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
            with open(outfile_flag,'w') as f:
                pass
            with h5py.File(outfile, 'w') as fw, h5py.File(f'{inpath}{file}') as fr, h5py.File(f'{outpathFullmom}{file}','w') as fw2:
                fw.create_dataset('notes',data=['time,mom,dirac',f'[time@fwd]=0:{tmax}; [time@bwd]=-{tmax}:-1','[dirac@fwd]=[0,1,4,5]; [dirac@bwd]=[10,11,14,15]'])
                fw.create_dataset('moms',data=moms_N)
                
                moms=[list(mom) for mom in fr['mvec']]
                momMap=np.array([moms.index(mom) for mom in moms_N])

                srcs=[]; datDic={'N1_N1':0,'N2_N2':0}
                for src in fr['baryons/nucl_nucl/twop_baryon_1'].keys():
                    (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                    (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                    src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                    srcs.append(src_new)
                    
                    for ky in ['twop_baryon_1','twop_baryon_2']:
                        ky_new={'twop_baryon_1':'N1_N1','twop_baryon_2':'N2_N2'}[ky]
                        tF=fr['baryons/nucl_nucl'][ky][src]
                        t=tF[...,0]+1j*tF[...,1]
                        datDic[ky_new]+=t
                        t=t[:,momMap,:]
                        fw.create_dataset('data/'+src_new+'/'+ky_new,data=t[:tmax+1,:,[0,1,4,5]])
                        fw.create_dataset('data_bw/'+src_new+'/'+ky_new,data=t[-tmax:,:,[10,11,14,15]])
                
                fw2.create_dataset('notes',data=['time,mom,dirac',f'[time@fwd]=0:{tmax}; [time@bwd]=-{tmax}:-1','[dirac@fwd]=[0,5]; [dirac@bwd]=[10,15]'])
                fw2.create_dataset('moms',data=moms)
                fw2.create_dataset('srcs',data=srcs)
                for ky_new in datDic.keys():
                    t=datDic[ky_new]/len(srcs)
                    fw2.create_dataset(f'data/{ky_new}',data=t[:tmax+1,:,[0,5]])
                    fw2.create_dataset(f'data_bw/{ky_new}',data=t[-tmax:,:,[10,15]])
            os.remove(outfile_flag)
            
    
    file="N.h5_nucl_all_source"
    if True:
        outfile=f'{outpath}{file}'
        outfile_flag=outfile+'_flag'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
            with open(outfile_flag,'w') as f:
                pass
            with h5py.File(outfile, 'w') as fw, h5py.File(f'{outpathFullmom}{file}','w') as fw2:
                fw.create_dataset('notes',data=['time,mom,dirac',f'[time@fwd]=0:{tmax}; [time@bwd]=-{tmax}:-1','[dirac@fwd]=[0,1,4,5]; [dirac@bwd]=[10,11,14,15]'])
                fw.create_dataset('moms',data=moms_N)
                
                tar = tarfile.open(inpath2)
                for tarinfo in tar:
                    handle = tar.extractfile(tarinfo)
                    t_cfg_old = tarinfo.name.split('/')[1].split('.')[0]
                    if t_cfg_old != cfg_old:
                        continue
                    with h5py.File(handle) as fr:
                        moms=fr['mvec']
                        momDic={}
                        for i,mom in enumerate(moms):
                            momDic[tuple(mom)]=i
                        momMap=np.array([momDic[tuple(mom)] for mom in moms_N])

                        srcs=[]; datDic={'N1_N1':0,'N2_N2':0}
                        for src in fr['baryons/nucl_nucl/twop_baryon_1'].keys():
                            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                            src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                            srcs.append(src_new)

                            for ky in ['twop_baryon_1','twop_baryon_2']:
                                ky_new={'twop_baryon_1':'N1_N1','twop_baryon_2':'N2_N2'}[ky]
                                tF=fr['baryons/nucl_nucl'][ky][src][:]
                                t=tF[...,0]+1j*tF[...,1]
                                datDic[ky_new]+=t
                                t=t[:,momMap,:]
                                fw.create_dataset('data/'+src_new+'/'+ky_new,data=t[:tmax+1,:,[0,1,4,5]])
                                fw.create_dataset('data_bw/'+src_new+'/'+ky_new,data=t[-tmax:,:,[10,11,14,15]])
                                
                        fw2.create_dataset('notes',data=['time,mom,dirac',f'[time@fwd]=0:{tmax}; [time@bwd]=-{tmax}:-1','[dirac@fwd]=[0,5]; [dirac@bwd]=[10,15]'])
                        fw2.create_dataset('moms',data=moms)
                        fw2.create_dataset('srcs',data=srcs)
                        for ky_new in datDic.keys():
                            t=datDic[ky_new]/len(srcs)
                            fw2.create_dataset(f'data/{ky_new}',data=t[:tmax+1,:,[0,5]])
                            fw2.create_dataset(f'data_bw/{ky_new}',data=t[-tmax:,:,[10,15]])
            os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)
    
run()