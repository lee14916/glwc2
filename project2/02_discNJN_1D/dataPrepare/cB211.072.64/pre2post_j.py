# cyclone
'''
cat data_aux/cfgs_all_j | xargs -I @ -P 10 python3 -u pre2post_j.py -c @ > log/pre2post_j.out & 
'''
import re, click
import h5py, os
import numpy as np

ens='cB211.072.64'
MUL=0.00072
MUS=0.0186
MUC=0.249
KAPPA=0.1394265

basePath=f'/nvme/h/cy22yl1/projectData/02_discNJN_1D/{ens}/'

cfg2old=lambda cfg: cfg[1:]+'_r'+{'a':'0','b':'1','c':'2','d':'3'}[cfg[0]]
cfg2new=lambda cfg: {'0':'a','1':'b','2':'c','3':'d'}[cfg[-1]] + cfg[:4]

# pf1 pf2 pc pi1 pi2
Nmax={'cB211.072.64':23,'cC211.060.80':26,'cD211.054.96':26}[ens]
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
base_momList=[[x,y,z] for x in t_range for y in t_range for z in t_range if np.linalg.norm([x,y,z])**2<=Nmax]
base_momList.sort()
target_momList=[mom for mom in base_momList]
target_momList.sort()

gamma_1=gamma_x=np.array([[0.,0.,0.,1j],[0.,0.,1j,0.],[0.,-1j,0.,0.],[-1j,0.,0.,0.]])
gamma_2=gamma_y=np.array([[0.,0.,0.,1.],[0.,0.,-1.,0.],[0.,-1.,0.,0.],[1.,0.,0.,0.]])
gamma_3=gamma_z=np.array([[0.,0.,1j,0.],[0.,0.,0.,-1j],[-1j,0.,0.,0.],[0.,1j,0.,0.]])
gamma_4=gamma_t=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,-1.]])
gamma_5=(gamma_1@gamma_2@gamma_3@gamma_4)

gms=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']

t=1/2
gmDic={
    'id':np.eye(4), 'gx':gamma_1, 'gy':gamma_2, 'gz':gamma_3, 'gt':gamma_4, 'g5':gamma_5,
    'g5gx':gamma_5@gamma_1, 'g5gy':gamma_5@gamma_2, 'g5gz':gamma_5@gamma_3, 'g5gt':gamma_5@gamma_4,
    'sgmxy':(gamma_x@gamma_y-gamma_y@gamma_x)*t,'sgmyz':(gamma_y@gamma_z-gamma_z@gamma_y)*t,'sgmzx':(gamma_z@gamma_x-gamma_x@gamma_z)*t,
    'sgmtx':(gamma_t@gamma_x-gamma_x@gamma_t)*t,'sgmty':(gamma_t@gamma_y-gamma_y@gamma_t)*t,'sgmtz':(gamma_t@gamma_z-gamma_z@gamma_t)*t 
}
signlessClass_g5comu=['id','g5','sgmxy','sgmyz','sgmzx','sgmtx','sgmty','sgmtz']

gmArray_p_std=np.array([1j*gamma_5@gmDic[gm] if gm in signlessClass_g5comu else np.zeros([4,4]) for gm in gms])
gmArray_p_gen=np.array([gmDic[gm] if gm not in signlessClass_g5comu else np.zeros([4,4]) for gm in gms])
gmArray_m_std=np.array([gmDic[gm] if gm not in signlessClass_g5comu else np.zeros([4,4]) for gm in gms])
gmArray_m_gen=np.array([1j*gamma_5@gmDic[gm] if gm in signlessClass_g5comu else np.zeros([4,4]) for gm in gms])

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    cfg_old=cfg2old(cfg)
    inpath=f'{basePath}data_pre_j/{cfg_old}/'
    outpath=f'{basePath}data_post/{cfg}/'

    files=os.listdir(inpath)
    
    os.makedirs(outpath,exist_ok=True)
    outfile=f'{outpath}j.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile, 'w') as fw:
            fw.create_dataset('moms',data=target_momList)
            fw.create_dataset('inserts',data=gms)
            if 'j.h5_exact' in files and 'j.h5_stoch' in files:
                for case in ['local','d0','d1','d2','d3']:
                    case2={'local':'local','d0':'dir_00','d1':'dir_01','d2':'dir_02','d3':'dir_03'}[case]
                    case3={'local':'local','d0':'dir0','d1':'dir1','d2':'dir2','d3':'dir3'}[case]
                    
                    t_std=t_gen=0
                    with h5py.File(inpath+'j.h5_exact') as fe, h5py.File(inpath+'j.h5_stoch') as fs:
                        Ndivide=1*512
                        ky_cfg='conf_'+cfg[1:]
                                
                        moms=fe['Momenta_list_xyz']
                        momDic={}
                        for i,mom in enumerate(moms):
                            momDic[tuple(mom)]=i
                        momMap=[momDic[tuple(mom)] for mom in target_momList]
                        
                        if case in ['local']:
                            t=fe[ky_cfg]['Scalar']['loop'][:] + fs[ky_cfg]['Nstoch_0001']['Scalar']['loop'][:]/Ndivide
                        else:
                            t=fe[ky_cfg]['Loops'][case2]['loop'][:] + fs[ky_cfg]['Nstoch_0001']['Loops'][case2]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,momMap]
                        t=np.reshape(t,[t.shape[0],-1,4,4])
                        t_std+=t
                        
                        if case in ['local']:
                            t=fe[ky_cfg]['dOp']['loop'][:] + fs[ky_cfg]['Nstoch_0001']['dOp']['loop'][:]/Ndivide
                        else:
                            t=fe[ky_cfg]['LpsDw'][case2]['loop'][:] + fs[ky_cfg]['Nstoch_0001']['LpsDw'][case2]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,momMap]
                        t=np.reshape(t,[t.shape[0],-1,4,4])
                        t_gen+=t
                        
                    # print(t_std.shape) # targe: t,m,a,b
                    
                    N_S=1
                    t_std=t_std*(-8*1j*MUL*KAPPA**2)/N_S
                    t_gen=t_gen*(-4*KAPPA)/N_S

                    t_p=np.einsum('gab,tmab->tmg',gmArray_p_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_p_gen,t_gen)
                    t_m=np.einsum('gab,tmab->tmg',gmArray_m_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_m_gen,t_gen)
                    
                    label='' if case in ['local'] else ';'+case
                    fw.create_dataset('data/j+'+label,data=t_p)
                    fw.create_dataset('data/j-'+label,data=t_m)
                
            if 'js.h5_stoch_D8' in files and 'js.h5_stoch_gen_D8_S2' in files and 'js.h5_stoch_std_D8_S2' in files:
                for case in ['local','d0','d1','d2','d3']:
                    case2={'local':'local','d0':'dir_00','d1':'dir_01','d2':'dir_02','d3':'dir_03'}[case]
                    case3={'local':'local','d0':'dir0','d1':'dir1','d2':'dir2','d3':'dir3'}[case]
                    
                    N_S=0
                    t_std=t_gen=0
                    with h5py.File(inpath+'js.h5_stoch_D8') as fs:
                        N_S+=1
                        Ndivide=1*512
                        ky_cfg='conf_'+cfg[1:]
                    
                        moms=fs['Momenta_list_xyz']
                        momDic={}
                        for i,mom in enumerate(moms):
                            momDic[tuple(mom)]=i
                        momMap=[momDic[tuple(mom)] for mom in target_momList]
                        
                        if case in ['local']:
                            t=fs[ky_cfg]['Nstoch_0001']['Scalar']['loop'][:]/Ndivide
                        else:
                            t=fs[ky_cfg]['Nstoch_0001']['Loops'][case2]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,momMap]
                        t=np.reshape(t,[t.shape[0],-1,4,4])
                        t_std+=t
                        
                        if case in ['local']:
                            t=fs[ky_cfg]['Nstoch_0001']['dOp']['loop'][:]/Ndivide
                        else:
                            t=fs[ky_cfg]['Nstoch_0001']['LpsDw'][case2]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,momMap]
                        t=np.reshape(t,[t.shape[0],-1,4,4])
                        t_gen+=t
                        
                    with h5py.File(inpath+'js.h5_stoch_std_D8_S2') as fss, h5py.File(inpath+'js.h5_stoch_gen_D8_S2') as fsg:
                        N_S+=1
                        Ndivide=1*512
                        ky_cfg='Conf'+cfg_old
                        
                        if case in ['local']:
                            moms=fss[ky_cfg]['Ns0']['localLoops']['mvec']
                        else:
                            moms=fss[ky_cfg]['Ns0']['oneD'][case3]['mvec']
                        momDic={}
                        for i,mom in enumerate(moms):
                            momDic[tuple(mom)]=i
                        momMap=[momDic[tuple(mom)] for mom in target_momList]

                        if case in ['local']:
                            t=fss[ky_cfg]['Ns0']['localLoops']['loop'][:]/Ndivide
                        else:
                            t=fss[ky_cfg]['Ns0']['oneD'][case3]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=np.transpose(t,[0,3,1,2])
                        t=t[:,momMap]
                        t_std+=t
                        
                        if case in ['local']:
                            t=fsg[ky_cfg]['Ns0']['localLoops']['loop'][:]/Ndivide
                        else:
                            t=fsg[ky_cfg]['Ns0']['oneD'][case3]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=np.transpose(t,[0,3,1,2])
                        t=t[:,momMap]
                        t_gen+=t
                    
                    t_std=t_std*(-8*1j*MUS*KAPPA**2)/N_S
                    t_gen=t_gen*(-4*KAPPA)/N_S

                    t_p=np.einsum('gab,tmab->tmg',gmArray_p_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_p_gen,t_gen)
                    t_m=np.einsum('gab,tmab->tmg',gmArray_m_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_m_gen,t_gen)
                    
                    label='' if case in ['local'] else ';'+case
                    fw.create_dataset('data/js'+label,data=t_p/2)
                
            if 'jc.h5_stoch' in files:
                for case in ['local','d0','d1','d2','d3']:
                    case2={'local':'local','d0':'dir_00','d1':'dir_01','d2':'dir_02','d3':'dir_03'}[case]
                    case3={'local':'local','d0':'dir0','d1':'dir1','d2':'dir2','d3':'dir3'}[case]
                    
                    t_std=t_gen=0
                    with h5py.File(inpath+'jc.h5_stoch') as fs:
                        Ndivide=12*32
                        ky_cfg='conf_'+cfg[1:]
                    
                        moms=fs['Momenta_list_xyz']
                        momDic={}
                        for i,mom in enumerate(moms):
                            momDic[tuple(mom)]=i
                        momMap=[momDic[tuple(mom)] for mom in target_momList]
                        
                        if case in ['local']:
                            t=fs[ky_cfg]['Nstoch_0012']['Scalar']['loop'][:]/Ndivide
                        else:
                            t=fs[ky_cfg]['Nstoch_0012']['Loops'][case2]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,momMap]
                        t=np.reshape(t,[t.shape[0],-1,4,4])
                        t_std+=t
                        
                        if case in ['local']:
                            t=fs[ky_cfg]['Nstoch_0012']['dOp']['loop'][:]/Ndivide
                        else:
                            t=fs[ky_cfg]['Nstoch_0012']['LpsDw'][case2]['loop'][:]/Ndivide
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,momMap]
                        t=np.reshape(t,[t.shape[0],-1,4,4])
                        t_gen+=t
                        
                    N_S=1
                    t_std=t_std*(-8*1j*MUC*KAPPA**2)/N_S
                    t_gen=t_gen*(-4*KAPPA)/N_S

                    t_p=np.einsum('gab,tmab->tmg',gmArray_p_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_p_gen,t_gen)
                    t_m=np.einsum('gab,tmab->tmg',gmArray_m_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_m_gen,t_gen)
                    
                    label='' if case in ['local'] else ';'+case
                    fw.create_dataset('data/jc'+label,data=t_p/2)
            
        os.remove(outfile_flag)
    print('flag_cfg_done: '+cfg)
        
run()