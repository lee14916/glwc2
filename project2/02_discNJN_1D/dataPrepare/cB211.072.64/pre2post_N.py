'''
cat data_aux/cfgs_all_N | xargs -I @ -P 10 python3 -u pre2post_N.py -c @ > log/pre2post_N.out & 
'''

import re, click
import h5py, os, tarfile
import numpy as np

basePath='/p/project/ngff/li47/code/projectData/02_discNJN_1D/cB211.072.64/'

cfg2old=lambda cfg: cfg[1:]+'_r'+{'a':'0','b':'1','c':'2','d':'3'}[cfg[0]]
cfg2new=lambda cfg: {'0':'a','1':'b','2':'c','3':'d'}[cfg[-1]] + cfg[:4]

# pf1 pf2 pc pi1 pi2
Nmax=4
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
base_momList=[[x,y,z] for x in t_range for y in t_range for z in t_range if np.linalg.norm([x,y,z])**2<=Nmax]
base_momList.sort()
target_momList=[mom for mom in base_momList]
target_momList.sort()
target_momList=np.array(target_momList)

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    cfg_old=cfg2old(cfg)
    inpath=f'{basePath}data_pre_N/{cfg_old}/'
    outpath=f'{basePath}data_post/{cfg}/'
    outpathFullmom=f'{basePath}data_N_fullmom/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    os.makedirs(outpathFullmom,exist_ok=True)
    
    files=os.listdir(inpath)
    
    file='N.h5_twop_threep_dt20_64srcs'
    if file in files:
        outfile=f'{outpath}{file}'
        outfile_flag=outfile+'_flag'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
            with open(outfile_flag,'w') as f:
                pass
            with h5py.File(outfile, 'w') as fw, h5py.File(f'{inpath}{file}') as fr, h5py.File(f'{outpathFullmom}{file}','w') as fw2:
                fw.create_dataset('moms',data=target_momList)
                moms=fr['mvec']
                momDic={}
                for i,mom in enumerate(moms):
                    momDic[tuple(mom)]=i
                momMap=np.array([momDic[tuple(mom)] for mom in target_momList])

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
                        fw.create_dataset('data/'+src_new+'/'+ky_new,data=t)
                        
                fw2.create_dataset('moms',data=fr['mvec'])
                fw2.create_dataset('srcs',data=srcs)
                for ky_new in datDic.keys():
                    fw2.create_dataset(f'data/{ky_new}',data=datDic[ky_new]/len(srcs))
            os.remove(outfile_flag)
        
    file='N.h5_twop_threep_2'
    if file in files:
        outfile=f'{outpath}{file}'
        outfile_flag=outfile+'_flag'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
            with open(outfile_flag,'w') as f:
                pass
            with h5py.File(outfile, 'w') as fw, h5py.File(f'{inpath}{file}') as fr, h5py.File(f'{outpathFullmom}{file}','w') as fw2:
                fw.create_dataset('moms',data=target_momList)
                moms=fr['mvec']
                momDic={}
                for i,mom in enumerate(moms):
                    momDic[tuple(mom)]=i
                momMap=np.array([momDic[tuple(mom)] for mom in target_momList])

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
                        fw.create_dataset('data/'+src_new+'/'+ky_new,data=t)
                        
                fw2.create_dataset('moms',data=fr['mvec'])
                fw2.create_dataset('srcs',data=srcs)
                for ky_new in datDic.keys():
                    fw2.create_dataset(f'data/{ky_new}',data=datDic[ky_new]/len(srcs))
            os.remove(outfile_flag)
        
    file='N.h5_hch02k_twop'
    if file in files:
        outfile=f'{outpath}{file}'
        outfile_flag=outfile+'_flag'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
            with open(outfile_flag,'w') as f:
                pass
            with h5py.File(outfile, 'w') as fw, h5py.File(f'{outpathFullmom}{file}','w') as fw2:
                fw.create_dataset('moms',data=target_momList)
                tar = tarfile.open(f'{inpath}{file}')
                moms2=None
                srcs=[]; datDic={'N1_N1':0,'N2_N2':0}
                for tarinfo in tar:
                    if 'baryons' not in tarinfo.name:
                        continue
                    handle = tar.extractfile(tarinfo)
                    with h5py.File(handle) as fr:
                        moms=fr['Momenta_list_xyz']
                        if moms2 is None:
                            moms2=fr['Momenta_list_xyz'][:]
                            assert(np.all(moms2==moms))
                        momDic={}
                        for i,mom in enumerate(moms):
                            momDic[tuple(mom)]=i
                        momMap=np.array([momDic[tuple(mom)] for mom in target_momList])
                        ky_cfg='conf_'+cfg[1:]
                        
                        for src in fr[ky_cfg].keys():
                            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                            src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                            srcs.append(src_new)
                            
                            for ky in ['twop_baryon_1','twop_baryon_2']:
                                ky_new={'twop_baryon_1':'N1_N1','twop_baryon_2':'N2_N2'}[ky]
                                t=fr[ky_cfg][src]['nucl_nucl'][ky][:]
                                t=t[...,0]+1j*t[...,1]
                                datDic[ky_new]+=t
                                t=t[:,momMap,:]
                                fw.create_dataset('data/'+src_new+'/'+ky_new,data=t)

                fw2.create_dataset('moms',data=moms2)
                fw2.create_dataset('srcs',data=srcs)
                for ky_new in datDic.keys():
                    fw2.create_dataset(f'data/{ky_new}',data=datDic[ky_new]/len(srcs))
            os.remove(outfile_flag)

    print('flag_cfg_done: '+cfg)
    
run()