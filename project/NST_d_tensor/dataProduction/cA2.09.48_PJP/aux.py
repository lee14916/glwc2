import math
import numpy as np
import pickle, os
from scipy.sparse import csr_matrix

def cfg2post(cfg,case):
    file={'pi0f':'pi0f.h5_loops-a-Nstoc100','P':'P.h5_jPP-a-tensor',
            'j':'j.h5_loops-a-Nstoc400','jPi':'jPi.h5_jPP-a-tensor',
            'PJP':'PJP.h5_PJP-a'}[case]
    return 'data_post/'+cfg+'/'+file

tfList=[10,12,14]
gjList=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']
# gjList=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt']
Format_ab=2 # 2 (4 eles) or 4 (16 eles) 

Psgn={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':-1,'sgmxy':1,'sgmyz':1,'sgmzx':1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}
PTsgn={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':-1,'g5':1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':-1,'sgmxy':1,'sgmyz':1,'sgmzx':1,'sgmtx':1,'sgmty':1,'sgmtz':1} # PT transformation acting on insertion
gtCj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':1,'sgmty':1,'sgmtz':1}
g5Cj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':-1,'g5':1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}

sgn_P=np.array([Psgn[gm] for gm in gjList])
sgn_PT=np.array([PTsgn[gm] for gm in gjList])
ens='cA2.09.48'
if ens in ['cA211.530.24']:
    (lat_L,lat_T)=(24,48)
    Tpack=16
    Npack=lat_T//Tpack
elif ens in ['cA2.09.48']:
    (lat_L,lat_T)=(48,96)
    Tpack=24
    Npack=lat_T//Tpack
    
path_cfgs='data_aux/cfgs_run'
path_diags_all='data_aux/diags_all'
path_diags_main='data_aux/diags_main'
path_group_coeffs='data_aux/group_coeffs.pkl'
path_opabsDic='data_aux/opabsDic.pkl'
path_auxDic='data_aux/auxDic.pkl'

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
        [[],['PJP']]],
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
    '''
    out -> post -> avgsrc ( -> avgmore ) -> merge

    # post -> avgsrc
    post: moms
    pavg (pre-avg): t-basis
    avgsrc: g-basis
    
    '''
    
    os.makedirs('log', exist_ok=True)
    
    with open(path_diags_all,'w') as f:
        f.write('\n'.join(diags_all))
    with open(path_diags_main,'w') as f:
        f.write('\n'.join(diags_main))        
    
    with open(path_group_coeffs,'rb') as f:
        coeffs=pickle.load(f)
    
    # all the ops in group basis required
    max_mom2=1
    momDic={}
    for mom2 in range(max_mom2+1):
        momDic[mom2]=[]
    t=math.floor(np.sqrt(max_mom2))
    for x in range(-t,t+1):
        for y in range(-t,t+1):
            for z in range(-t,t+1):
                mom2=x**2+y**2+z**2
                if mom2 in momDic:
                    momDic[mom2].append([x,y,z])
    ops=[]                
    for l in ['l1','l2']:
        for mom2,moms in momDic.items():
            for x,y,z in moms:
                op_base='g;{},{},{};'.format(x,y,z)
                op_base+='{};'.format('G1g' if mom2==0 else 'G1' if mom2 in [1,4] else 'G' if mom2 in [2,3] else 'NA')
                op=op_base+'a;{};N'.format(l); ops.append(op)
                occs = {0:['N1pi1,a'],1:['N1pi0,a','N0pi1,a']}[mom2]
                for occ in occs:
                    op=op_base+'{};{};N,pi'.format(occ,l); ops.append(op)
    ops+=['g;0,0,0;G1u;N0pi0,a;l1;N,pi','g;0,0,0;G1u;N0pi0,a;l2;N,pi']
    ops.sort()    
    
    opabsDic={'post':{},'pavg':{},'avg':{}}
    # avg
    for opa in ops:
        for opb in ops:
            _,moma,irrepa,_,la,flaa=opa.split(';')
            _,momb,irrepb,_,lb,flab=opb.split(';')
            t1={'N':{'N'},'N,pi':{'N','pia'}}[flaa]
            t2={'N':{'N'},'N,pi':{'N','pib'}}[flab]
            opab=opa+'_'+opb
            if (moma,irrepa,la)==(momb,irrepb,lb):
                dgtp=set2key(t1|t2)
                if dgtp not in opabsDic['avg']:
                    opabsDic['avg'][dgtp]=[]
                opabsDic['avg'][dgtp].append(opab)
            if np.sum(np.array(opab2pc(opab))**2)<=max_mom2:
                dgtp=set2key(t1|t2|{'j'})
                if dgtp not in opabsDic['avg']:
                    opabsDic['avg'][dgtp]=[]
                opabsDic['avg'][dgtp].append(opa+'_'+opb)
    for ky in opabsDic['avg']:
        opabsDic['avg'][ky]=list(set(opabsDic['avg'][ky]))
        opabsDic['avg'][ky].sort()
    flag_check=True # check if one off contains the other
    if flag_check:
        opabs=opabsDic['avg'][set2key({'N','pia'})]
        opabs=[opab.split('_')[1]+'_'+opab.split('_')[0] for opab in opabs]
        opabs.sort()
        assert(opabs==opabsDic['avg'][set2key({'N','pib'})])
        opabs=opabsDic['avg'][set2key({'N','pia','j'})]
        opabs=[opab.split('_')[1]+'_'+opab.split('_')[0] for opab in opabs]
        opabs.sort()
        assert(opabs==opabsDic['avg'][set2key({'N','pib','j'})])
    ky_dgtp_all=list(opabsDic['avg'].keys())
    ky_dgtp_extend=[set2key(dgtp) for dgtp in [{'pia'},{'pib'},{'j'},{'pia','pib'},{'pia','j'},{'j','pib'},{'pia','j','pib'}]]
    # pavg
    opabsDic['pavg']={}
    for ky in ky_dgtp_all:
        opabs=opabsDic['avg'][ky]
        opabs=[elea[0]+'_'+eleb[0] for opab in opabs for elea in coeffs[opab.split('_')[0]] for eleb in coeffs[opab.split('_')[1]]]
        opabs=list(set(opabs))
        moms=moms_unique([opab2mom(opab)[0] for opab in opabs])
        opabsDic['pavg'][ky]=moms
    # post
    for ky_base in ky_dgtp_all+ky_dgtp_extend:
        t_moms=[]
        for ky in ky_dgtp_all:
            if not set(ky_base).issubset(set(ky)):
                continue
            # if set(ky) in [{'N','pia'},{'N','j','pia'},{'N','j','pia','pib'}]:
            #     continue
            # if set(ky) in [{'N','j','pia','pib'}]:
            #     continue
            t_moms+=list(moms_full2base(opabsDic['pavg'][ky],set(ky_base)))
        opabsDic['post'][ky_base]=moms_unique(t_moms)
        # print(ky_base,len(opabsDic['post'][ky_base]))
        # print(opabsDic['post'][ky_base])
    flag_check=True # check if one off contains the other
    if flag_check:
        moms1=np.array(opabsDic['post'][set2key({'pia'})])[:,op2momind['pia']]
        moms2=np.array(opabsDic['post'][set2key({'pib'})])[:,op2momind['pib']]
        assert(np.all(moms1==moms2))
        moms1=[
            list(np.array(mom)[op2momind['pia']])+list(-np.array(mom)[op2momind['j']])
            for mom in opabsDic['post'][set2key({'pia','j'})]
        ]
        moms1.sort()
        moms1=np.array(moms1)
        moms2=np.array(opabsDic['post'][set2key({'pib','j'})])[:,op2momind['pib']+op2momind['j']]
        assert(np.all(moms1==moms2))
    # finalize opabsDic
    with open(path_opabsDic,'wb') as f:
        pickle.dump(opabsDic,f)
        
    # auxDic
    auxDic={'post2pavg':{},'pavg2avg':{}}
    # post -> pavg
    for ky_base in ky_dgtp_all+ky_dgtp_extend:
        for ky in ky_dgtp_all:
            pass
    # pavg -> avg
    for ky in ky_dgtp_all:
        moms=opabsDic['pavg'][ky]
        opabs=opabsDic['avg'][ky]
        dic=moms2dic(moms)
        data=[];row=[];col=[]
        for i,opab in enumerate(opabs):
            opa,opb=opab.split('_')
            for ele_a in coeffs[opa]:
                for ele_b in coeffs[opb]:
                    data.append(ele_a[1]*np.conj(ele_b[1]))
                    opab2mom(ele_a[0]+'_'+ele_b[0])
                    mom,ab=opab2mom(ele_a[0]+'_'+ele_b[0])
                    row.append(i); col.append(dic[tuple(mom)]*(Format_ab**2)+ab)
        auxDic['pavg2avg'][ky]=csr_matrix((data, (row, col)), shape=(len(opabs), len(moms)*(Format_ab**2)))
        # print(auxDic['pavg2avg'][ky].shape,auxDic['pavg2avg'][ky].getnnz())
    # finalize auxDic
    with open(path_auxDic,'wb') as f:
        pickle.dump(auxDic,f)