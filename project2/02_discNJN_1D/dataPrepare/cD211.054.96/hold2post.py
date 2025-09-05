'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u hold2post.py -c @ > log/hold2post.out & 
'''
import re, click
import h5py, os
import numpy as np

ens='cD211.054.96'
tmax={'cB211.072.64':36,'cC211.060.80':40,'cD211.054.96':48}[ens]

flags={
    'g5H_local':True,
}

#=========================================================================================

Nmax=0
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
moms_N=[[x,y,z] for x in t_range for y in t_range for z in t_range if x**2+y**2+z**2<=Nmax]
moms_N.sort()
#
# Nmax={'cB211.072.64':23,'cC211.060.80':26,'cD211.054.96':26,'cE211.044.112':4}[ens]
Nmax={'cB211.072.64':14,'cC211.060.80':16,'cD211.054.96':16,'cE211.044.112':4}[ens]
# Nmax={'cB211.072.64':1,'cC211.060.80':1,'cD211.054.96':1,'cE211.044.112':1}[ens]
Nmax_sq=int(np.floor(np.sqrt(Nmax))); t_range=range(-Nmax_sq,Nmax_sq+1)
moms_j=[[x,y,z] for x in t_range for y in t_range for z in t_range if x**2+y**2+z**2<=Nmax]
moms_j.sort()

id=np.eye(4)
gamma_1=gamma_x=np.array([[0.,0.,0.,1j],[0.,0.,1j,0.],[0.,-1j,0.,0.],[-1j,0.,0.,0.]])
gamma_2=gamma_y=np.array([[0.,0.,0.,1.],[0.,0.,-1.,0.],[0.,-1.,0.,0.],[1.,0.,0.,0.]])
gamma_3=gamma_z=np.array([[0.,0.,1j,0.],[0.,0.,0.,-1j],[-1j,0.,0.,0.],[0.,1j,0.,0.]])
gamma_4=gamma_t=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,-1.]])
gamma_5=(gamma_1@gamma_2@gamma_3@gamma_4)
#
P0=(id+gamma_4)/4; Px=1j*gamma_5@gamma_1@P0; Py=1j*gamma_5@gamma_2@P0; Pz=1j*gamma_5@gamma_3@P0
P0n=(id-gamma_4)/4; Pxn=1j*gamma_5@gamma_1@P0n; Pyn=1j*gamma_5@gamma_2@P0n; Pzn=1j*gamma_5@gamma_3@P0n
#
# coeff_{i} C_{i=4a+b} = Tr[proj@C] = proj_{ba} C_{ab} = proj_{ba} C_{4a+b} 
# coeff_{i=4a+b} = proj_{ba}
dirac2proj=np.array([[complex(ele) for row in proj.T for ele in row] for proj in [P0,Px,Py,Pz]])[:,[0,1,4,5]]
dirac2proj_bw=np.array([[complex(ele) for row in proj.T for ele in row] for proj in [P0n,Pxn,Pyn,Pzn]])[:,[10,11,14,15]]

# stouts=range(0,41)
stouts=[4,7,10,13,16,19,22]

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post_hold/{cfg}/'
    outpath=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    
    files=os.listdir(inpath)
    files_N=[file for file in files if file.startswith('N.h5')]
    for file in files_N:
        outfile=f'{outpath}{file}'
        outfile_flag=outfile+'_flag'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
            with open(outfile_flag,'w') as f:
                pass
            with h5py.File(outfile,'w') as fw, h5py.File(f'{inpath}{file}') as fr:
                fw.create_dataset('notes',data=['time,mom,proj',f'[time@fwd]=0:{tmax}; [time@bwd]=-{tmax}:-1','[proj]=[P0,Px,Py,Pz]'])
                fw.create_dataset('moms',data=moms_N)
                moms=[list(mom) for mom in fr['moms'][:]]
                momMap=[moms.index(mom) for mom in moms_N]
                
                for src in fr['data'].keys():
                    t=(fr[f'data/{src}/N1_N1'][:,momMap] + fr[f'data/{src}/N2_N2'][:,momMap])/2
                    t=np.einsum('pd,tmd->tmp',dirac2proj,t)
                    fw.create_dataset(f'data/{src}/N_N',data=t)
                    
                    t=(fr[f'data_bw/{src}/N1_N1'][:,momMap] + fr[f'data_bw/{src}/N2_N2'][:,momMap])/2
                    t=np.einsum('pd,tmd->tmp',dirac2proj_bw,t)
                    fw.create_dataset(f'data_bw/{src}/N_N',data=t)
                    
                    t=(fr[f'data/{src}/N1_N1'][:,momMap] - fr[f'data/{src}/N2_N2'][:,momMap])/2
                    t=np.einsum('pd,tmd->tmp',dirac2proj,t)
                    fw.create_dataset(f'data/{src}/N_N_-',data=t)
                    
                    t=(fr[f'data_bw/{src}/N1_N1'][:,momMap] - fr[f'data_bw/{src}/N2_N2'][:,momMap])/2
                    t=np.einsum('pd,tmd->tmp',dirac2proj_bw,t)
                    fw.create_dataset(f'data_bw/{src}/N_N_-',data=t)
                     
                    # break
            os.remove(outfile_flag)
            
    file='j.h5'; file2='jg.h5'
    outfile=f'{outpath}{file}'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw, h5py.File(f'{inpath}{file}') as fr, h5py.File(f'{inpath}{file2}') as fr2:
            fw.create_dataset('notes',data=['time,mom,insert','g{m,Dn} = (gamma_mu D_nu + gamma_nu D_mu)/2; tl=traceless', 'SGM[m{n],Dr} = sgm_', 'SGMmn=[gm,gn]/2'])
            fw.create_dataset('moms',data=moms_j)
            moms=[list(mom) for mom in fr['moms'][:]]
            momMap=[moms.index(mom) for mom in moms_j]
            
            gms=[gm.decode() for gm in fr['inserts'][:]]
            
            txyz=['t','x','y','z']; txyz2ind={'t':0,'x':1,'y':2,'z':3}
            insert2d={'x':'0','y':'1','z':'2','t':'3'}
            for ind_j,j in enumerate(['j+','j-','js','jc']):
                case='local'; inserts=gms
                if ind_j==0:
                    fw.create_dataset(f'inserts',data=inserts)
                t=fr[f'data/{j}'][:,momMap,:]
                if flags['g5H_local']:
                    sgn={'j+':1,'j-':-1,'js':1,'jc':1}[j]
                    g5Cj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':-1,'g5':1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}
                    sgnConj=np.array([g5Cj[gj] for gj in gms])
                    momMap_neg=np.array([moms.index(list(-np.array(mom))) for mom in moms_j])
                    t2=fr[f'data/{j}'][:,:]
                    t2=t2[:,momMap_neg]
                    t=(t+sgn*np.conj(t2)*sgnConj[None,None,:])/2
                fw.create_dataset(f'data/{j}',data=t)
                
                case='id,Dm'; inserts=txyz
                if ind_j==0:
                    fw.create_dataset(f'inserts;{case}',data=inserts)
                t=np.array([fr[f'data/{j};d{insert2d[m]}'][:,momMap,gms.index(f'id')] for m in inserts])
                t=np.transpose(t,[1,2,0])
                fw.create_dataset(f'data/{j};{case}',data=t)
                
                case='g5,Dm'; inserts=txyz
                if ind_j==0:
                    fw.create_dataset(f'inserts;{case}',data=inserts)
                t=np.array([fr[f'data/{j};d{insert2d[m]}'][:,momMap,gms.index(f'g5')] for m in inserts])
                t=np.transpose(t,[1,2,0])
                fw.create_dataset(f'data/{j};{case}',data=t)
                
                case='g{m,Dn};tl'; inserts=['tt','tx','ty','tz','xx','xy','xz','yy','yz','zz']
                if ind_j==0:
                    fw.create_dataset(f'inserts;{case}',data=inserts)
                t=np.array([[fr[f'data/{j};d{insert2d[n]}'][:,momMap,gms.index(f'g{m}')] for n in txyz] for m in txyz])
                t=(t+np.transpose(t,[1,0,2,3]))/2
                t=t - np.eye(4)[:,:,None,None]*np.trace(t,axis1=0,axis2=1)[None,None,:,:]/4
                t=[t[txyz.index(m),txyz.index(n)] for m,n in inserts]
                t=np.transpose(t,[1,2,0])
                fw.create_dataset(f'data/{j};{case}',data=t)
                
                # case='g5gm,Dn'; inserts=['tt','tx','ty','tz','xt','xx','xy','xz','yt','yx','yy','yz','zt','zx','zy','zz']
                # fw.create_dataset(f'inserts;{case}',data=inserts)
                # t=np.array([fr[f'data/{j};d{insert2d[n]}'][:,momMap,gms.index(f'g5g{m}')] for m,n in inserts])
                # t=np.transpose(t,[1,2,0])
                # fw.create_dataset(f'data/{j};{case}',data=t)
                #
                case='g5g{m,Dn};tl'; inserts=['tt','tx','ty','tz','xx','xy','xz','yy','yz','zz']
                if ind_j==0:
                    fw.create_dataset(f'inserts;{case}',data=inserts)
                t=np.array([[fr[f'data/{j};d{insert2d[n]}'][:,momMap,gms.index(f'g5g{m}')] for n in txyz] for m in txyz])
                t=(t+np.transpose(t,[1,0,2,3]))/2
                t=t - np.eye(4)[:,:,None,None]*np.trace(t,axis1=0,axis2=1)[None,None,:,:]/4
                t=[t[txyz.index(m),txyz.index(n)] for m,n in inserts]
                t=np.transpose(t,[1,2,0])
                fw.create_dataset(f'data/{j};{case}',data=t)
                #
                case='g5g[m,Dn]'; inserts=['tx','ty','tz','xy','xz','yz']
                if ind_j==0:
                    fw.create_dataset(f'inserts;{case}',data=inserts)
                t=np.array([[fr[f'data/{j};d{insert2d[n]}'][:,momMap,gms.index(f'g{m}')] for n in txyz] for m in txyz])
                t=(t-np.transpose(t,[1,0,2,3]))/2
                t=[t[txyz.index(m),txyz.index(n)] for m,n in inserts]
                t=np.transpose(t,[1,2,0])
                fw.create_dataset(f'data/{j};{case}',data=t)
                
                case='SGMmn,Dr'; inserts=['txt','tyt','tzt','xyt','xzt','yzt', 'txx','tyx','tzx','xyx','xzx','yzx', \
                    'txy','tyy','tzy','xyy','xzy','yzy', 'txz','tyz','tzz','xyz','xzz','yzz']
                if ind_j==0:
                    fw.create_dataset(f'inserts;{case}',data=inserts)
                def get(m,n,r):
                    if m==n:
                        return 0 * fr[f'data/{j};d{insert2d["t"]}'][:,momMap,gms.index(f'sgmxy')]
                    if f'sgm{m}{n}' in gms:
                        return fr[f'data/{j};d{insert2d[r]}'][:,momMap,gms.index(f'sgm{m}{n}')] 
                    return -fr[f'data/{j};d{insert2d[r]}'][:,momMap,gms.index(f'sgm{n}{m}')] 
                t=np.array([get(m,n,r) for m,n,r in inserts])
                t=np.transpose(t,[1,2,0])
                fw.create_dataset(f'data/{j};{case}',data=t)
                
                # case='SGM[m{n],Dr};tl'; inserts=['txx','txy','txz','tyy','tyz','tzz','xyy','xyz','yzz','yzz']
                # fw.create_dataset(f'inserts;{case}',data=inserts)
                # def get(m,n,r):
                #     if m==n:
                #         return 0 * fr[f'data/{j};d{insert2d["t"]}'][:,momMap,gms.index(f'sgmxy')]
                #     if f'sgm{m}{n}' in gms:
                #         return fr[f'data/{j};d{insert2d[r]}'][:,momMap,gms.index(f'sgm{m}{n}')] 
                #     return -fr[f'data/{j};d{insert2d[r]}'][:,momMap,gms.index(f'sgm{n}{m}')] 
                # t=np.array([[[get(m,n,r) for r in txyz] for n in txyz] for m in txyz])
                # t=np.array([t[i] - np.eye(4)[:,:,None,None]*np.trace(t[i],axis1=0,axis2=1)[None,None,:,:]/4 for i in range(4)])
                # orders=[[0,1,2],[0,2,1],[2,1,0],[1,0,2],[2,0,1],[1,2,0]]; sgns=[1,1,1,-1,-1,-1]
                # t=np.mean([sgn*np.transpose(t,order+[3,4]) for order,sgn in zip(orders,sgns)],axis=0)
                # # t=np.array([t[i] - np.eye(4)[:,:,None,None]*np.trace(t[i],axis1=0,axis2=1)[None,None,:,:]/4 for i in range(4)])
                # t=[t[txyz.index(m),txyz.index(n),txyz.index(r)] for m,n,r in inserts]
                # t=np.transpose(t,[1,2,0])
                # fw.create_dataset(f'data/{j};{case}',data=t)
                
                
            moms=[list(mom) for mom in fr2['moms'][:]]
            momMap=[moms.index(mom) for mom in moms_j]
            
            gms=[gm.decode() for gm in fr2['inserts'][:]]
            inserts=gms
            fw.create_dataset(f'inserts;jg',data=inserts)
            
            ind_diags=[inserts.index(i+i) for i in ['t','x','y','z']]
            
            for stout in stouts:
                j=f'jg;stout{stout}'
                t=fr2[f'data/{j}'][:]
                t=t[:,momMap,:]
                tr=np.sum(t[:,:,ind_diags],axis=2)
                t[:,:,ind_diags] -= tr[:,:,None]/4
                fw.create_dataset(f'data/{j}',data=t)
                
        os.remove(outfile_flag)
            
    print('flag_cfg_done: '+cfg)
            
run()