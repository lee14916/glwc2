import re
import h5py, os
import numpy as np

inpath='data_pre/'
outpath='data_post/'

# pf1 pf2 pc pi1 pi2
Nmax=4
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
base_momList=[[x,y,z] for x in t_range for y in t_range for z in t_range if np.linalg.norm([x,y,z])**2<=Nmax]
base_momList.sort()
target_momList=[mom for mom in base_momList]
target_momList.sort()

MUL=0.00072
MUS=0.0186
MUC=0.249
KAPPA=0.1394265

gamma_1=gamma_x=np.array([[0.,0.,0.,1j],[0.,0.,1j,0.],[0.,-1j,0.,0.],[-1j,0.,0.,0.]])
gamma_2=gamma_y=np.array([[0.,0.,0.,1.],[0.,0.,-1.,0.],[0.,-1.,0.,0.],[1.,0.,0.,0.]])
gamma_3=gamma_z=np.array([[0.,0.,1j,0.],[0.,0.,0.,-1j],[-1j,0.,0.,0.],[0.,1j,0.,0.]])
gamma_4=gamma_t=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,-1.]])
gamma_5=(gamma_1@gamma_2@gamma_3@gamma_4)

# gms=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmxy','sgmyz','sgmzx','sgmtx','sgmty','sgmtz']
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

# main
cfgs = [cfg for cfg in os.listdir(inpath)]
cfgs.sort()

for i_cfg,cfg in enumerate(cfgs):
    if i_cfg<=620:
        continue
    os.makedirs(outpath+cfg,exist_ok=True)
    with h5py.File(outpath+cfg+'/j.h5', 'w') as fw:
        fw.create_dataset('mvec',data=target_momList)

        # j
        t_std=t_gen=0
        with h5py.File(inpath+cfg+'/j.h5_exact') as fe, h5py.File(inpath+cfg+'/j.h5_stoch') as fs:
            Ndivide=1*512
            ky_cfg='conf_'+cfg[:-3]
                    
            moms=fe['Momenta_list_xyz']
            momDic={}
            for i,mom in enumerate(moms):
                momDic[tuple(mom)]=i
            momMap=[momDic[tuple(mom)] for mom in target_momList]
            
            t=fe[ky_cfg]['Scalar']['loop'][:] + fs[ky_cfg]['Nstoch_0001']['Scalar']['loop'][:]/Ndivide
            t=t[...,0]+1j*t[...,1]
            t=t[:,momMap]
            t=np.reshape(t,[t.shape[0],-1,4,4])
            t_std+=t
            
            t=fe[ky_cfg]['dOp']['loop'][:] + fs[ky_cfg]['Nstoch_0001']['dOp']['loop'][:]/Ndivide
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
        
        fw.create_dataset('data/j+',data=t_p)
        fw.create_dataset('data/j-',data=t_m)
        
        # js
        N_S=0
        t_std=t_gen=0
        with h5py.File(inpath+cfg+'/js.h5_stoch_D8') as fs:
            N_S+=1
            Ndivide=1*512
            ky_cfg='conf_'+cfg[:-3]
        
            moms=fs['Momenta_list_xyz']
            momDic={}
            for i,mom in enumerate(moms):
                momDic[tuple(mom)]=i
            momMap=[momDic[tuple(mom)] for mom in target_momList]
            
            t=fs[ky_cfg]['Nstoch_0001']['Scalar']['loop'][:]/Ndivide
            t=t[...,0]+1j*t[...,1]
            t=t[:,momMap]
            t=np.reshape(t,[t.shape[0],-1,4,4])
            t_std+=t
            
            t=fs[ky_cfg]['Nstoch_0001']['dOp']['loop'][:]/Ndivide
            t=t[...,0]+1j*t[...,1]
            t=t[:,momMap]
            t=np.reshape(t,[t.shape[0],-1,4,4])
            t_gen+=t
            
        with h5py.File(inpath+cfg+'/js.h5_stoch_std_D8_S2') as fss, h5py.File(inpath+cfg+'/js.h5_stoch_gen_D8_S2') as fsg:
            N_S+=1
            Ndivide=1*512
            ky_cfg='Conf'+cfg
            
            moms=fss[ky_cfg]['Ns0']['localLoops']['mvec']
            momDic={}
            for i,mom in enumerate(moms):
                momDic[tuple(mom)]=i
            momMap=[momDic[tuple(mom)] for mom in target_momList]

            t=fss[ky_cfg]['Ns0']['localLoops']['loop'][:]/Ndivide
            t=t[...,0]+1j*t[...,1]
            t=np.transpose(t,[0,3,1,2])
            t=t[:,momMap]
            t_std+=t
            
            t=fsg[ky_cfg]['Ns0']['localLoops']['loop'][:]/Ndivide
            t=t[...,0]+1j*t[...,1]
            t=np.transpose(t,[0,3,1,2])
            t=t[:,momMap]
            t_gen+=t
        
        t_std=t_std*(-8*1j*MUS*KAPPA**2)/N_S
        t_gen=t_gen*(-4*KAPPA)/N_S

        t_p=np.einsum('gab,tmab->tmg',gmArray_p_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_p_gen,t_gen)
        t_m=np.einsum('gab,tmab->tmg',gmArray_m_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_m_gen,t_gen)
        
        fw.create_dataset('data/js',data=t_p/2)
        
        # jc
        t_std=t_gen=0
        with h5py.File(inpath+cfg+'/jc.h5_stoch') as fs:
            Ndivide=12*32
            ky_cfg='conf_'+cfg[:-3]
        
            moms=fs['Momenta_list_xyz']
            momDic={}
            for i,mom in enumerate(moms):
                momDic[tuple(mom)]=i
            momMap=[momDic[tuple(mom)] for mom in target_momList]
            
            t=fs[ky_cfg]['Nstoch_0012']['Scalar']['loop'][:]/Ndivide
            t=t[...,0]+1j*t[...,1]
            t=t[:,momMap]
            t=np.reshape(t,[t.shape[0],-1,4,4])
            t_std+=t
            
            t=fs[ky_cfg]['Nstoch_0012']['dOp']['loop'][:]/Ndivide
            t=t[...,0]+1j*t[...,1]
            t=t[:,momMap]
            t=np.reshape(t,[t.shape[0],-1,4,4])
            t_gen+=t
            
        N_S=1
        t_std=t_std*(-8*1j*MUC*KAPPA**2)/N_S
        t_gen=t_gen*(-4*KAPPA)/N_S

        t_p=np.einsum('gab,tmab->tmg',gmArray_p_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_p_gen,t_gen)
        t_m=np.einsum('gab,tmab->tmg',gmArray_m_std,t_std)+np.einsum('gab,tmab->tmg',gmArray_m_gen,t_gen)
        
        fw.create_dataset('data/jc',data=t_p/2)
    
    print(i_cfg+1,len(cfgs),cfg,end='                                   \r')    
    # break
    