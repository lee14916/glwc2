import math
import numpy as np
import pickle, os
from scipy.sparse import csr_matrix

with open('data_aux/tfs','r') as f:
    tfs=f.read().splitlines()[0]
    tfList=[int(tf) for tf in tfs.split(',')]
tfList_disc=range(2,30+1)

pathBase='/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/01_Nsgm/cB211.072.64_base/'
temp=os.getcwd().split('/')[-1]
pathBaseTf=f'/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/01_Nsgm/cB211.072.64_base_{tfs}/'
pathBaseCode='/capstor/store/cscs/userlab/s1174/lyan/code/glwc2/project2/01_Nsgm/dataPrepare/cB211.072.64_base/'
os.makedirs(pathBase,exist_ok=True)
os.makedirs(pathBaseTf,exist_ok=True)

def cfg2post(cfg,case):
    file={'pi0f':'pi0f.h5_Nstoc25','P':'P.h5_jPP',
            'j':'j.h5_Nj','jPi':'jPi.h5_jPP'}[case]
    return f'{pathBaseTf}data_post/'+cfg+'/'+file

gjList=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']
# gjList=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt']
Format_ab=2 # 2 (4 eles) or 4 (16 eles) 

Psgn={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':-1,'sgmxy':1,'sgmyz':1,'sgmzx':1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}
PTsgn={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':-1,'g5':1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':-1,'sgmxy':1,'sgmyz':1,'sgmzx':1,'sgmtx':1,'sgmty':1,'sgmtz':1} # PT transformation acting on insertion
gtCj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':1,'sgmty':1,'sgmtz':1}
g5Cj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':-1,'g5':1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}

sgn_P=np.array([Psgn[gm] for gm in gjList])
sgn_PT=np.array([PTsgn[gm] for gm in gjList])
ens='cB211.072.64'
if ens in ['cA211.530.24']:
    (lat_L,lat_T)=(24,48)
    Tpack=16
    Npack=lat_T//Tpack
elif ens in ['cA2.09.48']:
    (lat_L,lat_T)=(48,96)
    Tpack=24
    Npack=lat_T//Tpack
elif ens in ['cB211.072.64']:
    (lat_L,lat_T)=(64,128)
    Tpack=32
    Npack=lat_T//Tpack
    
path_cfgs=pathBaseCode+'data_aux/cfgs_run'
path_diags_all=pathBase+'data_aux/diags_all'
path_diags_main=pathBase+'data_aux/diags_main'
path_group_coeffs=pathBaseCode+'data_aux/group_coeffs.pkl'
path_opabsDic=pathBase+'data_aux/opabsDic.pkl'
path_auxDic=pathBase+'data_aux/auxDic.pkl'

app_init=[['pi0i',{'pib'}],['pi0f',{'pia'}],['j',{'j'}],['P',{'pia','pib'}],\
    ['jPi',{'j','pib'}],['jPf',{'pia','j'}],['PJP',{'pia','j','pib'}]]
# diag_init=[
#     [['N','N_bw'],{'N'},\
#         [[],['pi0i'],['pi0f'],['P'],['pi0f','pi0i'], ['j'],['jPi'],['j','pi0i'],['jPf'],['pi0f','j'],
#      ['PJP'],['P','j'],['pi0f','jPi'],['jPf','pi0i'],['pi0f','j','pi0i']]],
#     [['T','T_bw'],{'N','pib'},\
#         [[],['pi0f'],['j'],['jPf'],['pi0f','j']]],
#     [['B2pt','W2pt','Z2pt','B2pt_bw','W2pt_bw','Z2pt_bw'],{'N','pia','pib'},\
#         [[],['j']]],
#     [['NJN'],{'N','j'},\
#         [[],['pi0i'],['pi0f'],['P'],['pi0f','pi0i']]],
#     [['B3pt','W3pt','Z3pt'],{'N','j','pib'},\
#         [[],['pi0f']]],
#     [['NpiJNpi'],{'N','pia','j','pib'},\
#         [[]]],
# ]
diag_init=[
    [['N','N_bw'],{'N'},\
        [[],['pi0i'],['pi0f'],['P'],['pi0f','pi0i'], ['j'],['jPi'],['j','pi0i'],['jPf'],['pi0f','j']]],
    [['T','T_bw'],{'N','pib'},\
        [[],['pi0f'],['j']]],
    [['B2pt','W2pt','Z2pt','B2pt_bw','W2pt_bw','Z2pt_bw'],{'N','pia','pib'},\
        [[]]],
    [['NJN'],{'N','j'},\
        [[],['pi0i'],['pi0f']]],
    [['B3pt','W3pt','Z3pt'],{'N','j','pib'},\
        [[]]],
    # [['NpiJNpi'],{'N','pia','j','pib'},\
    #     [[]]],
]
# for bases,base_dgtp,appss in diag_init:
#     print(bases,end=':\n\t')
#     for apps in appss:
#         print(apps,end=' ')
#     print()

diags_all=[]; diags_main=[]; diag2baps={}; diag2dgtp={} # baps=base+apps; dgtp=diagram type
for app,dgtp in app_init:
    diag2dgtp[app]=dgtp
for bases,base_dgtp,appss in diag_init:
    for base in bases:
        for apps in appss:
            diag='-'.join([base]+apps)
            diags_all.append(diag)
            diag2baps[diag]=(base,apps)
            diag2dgtp[diag]=set.union(*([base_dgtp]+[diag2dgtp[app] for app in apps]))
            # if diag2dgtp[diag] not in [{'N','pia'},{'N','j','pia'}]:
            if not base.endswith('_bw'):
                diags_main.append(diag)

op2momind={'N':[3,4,5],'pia':[6,7,8],'pib':[0,1,2],'j':[9,10,11]}
 
def set2key(s):
    t=list(s); t.sort()
    return tuple(t)
def op2pt(op):
    t=op.split(';')
    if t[0]=='t':
        t=t[1].split(',')
        t=[int(ele) for ele in t]
        if len(t)==3:
            return tuple(t)
        elif len(t)==6:
            return tuple([t[0]+t[3],t[1]+t[4],t[2]+t[5]])
    elif t[0]=='g':
        t=t[1].split(',')
        t=[int(ele) for ele in t]
        return tuple(t)
    1/0 
def opab2pc(opab):
    opa,opb=opab.split('_')
    pta=op2pt(opa); ptb=op2pt(opb)
    pc=np.array(ptb)-pta
    return pc
def opab2mom(opab):
    opa,opb=opab.split('_')
    _,moma,a,_=opa.split(';'); _,momb,b,_=opb.split(';')
    moma=[int(i) for i in moma.split(',')]; momb=[int(i) for i in momb.split(',')]
    moma+=[0,0,0] if len(moma)==3 else []; momb+=[0,0,0] if len(momb)==3 else [] 
    mom=momb[3:]+moma+list(opab2pc(opab))
    return mom,int(a)*Format_ab+int(b)
def moms_unique(moms):
    moms=list(set([tuple(mom) for mom in moms]))
    moms=[list(mom) for mom in moms]
    moms.sort()
    return moms
def moms_full2base(moms,dgtp):
    mominds=set([i for op in dgtp for i in op2momind[op]])
    filter=np.array([1 if i in mominds else 0 for i in range(12)])
    return moms*filter[None,:]
def moms2dic(moms):
    dic={}
    for i,ele in enumerate(moms):
        dic[tuple(ele)]=i
    return dic 


if __name__ == "__main__":
    os.makedirs('log', exist_ok=True)
    
    os.makedirs('data_aux', exist_ok=True)
    
    with open('data_aux/diags_all','w') as f:
        f.write('\n'.join(diags_all))
    with open('data_aux/diags_main','w') as f:
        f.write('\n'.join(diags_main))        