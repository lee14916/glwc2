'''
nohup python3 -u proj.py > log/proj.out &

For sigma, G1g and G1u are interchanged
'''

import os, h5py
from pandas import read_pickle
import numpy as np

# input
 
inPath='./avg/'
outPath='./proj/'
irrepCoefPath='./irrepCoef/'
irrepCoefPathN=irrepCoefPath+'NEWcoeffsN.pkl'
irrepCoefPathNpi=irrepCoefPath+'NEWcoeffsNpi.pkl'

patterns=[]

# patterns=['N_2pt','T','M_correct_2pt','B_2pt','W_2pt','Z_2pt','B','W','Z'] # for NJNpi_N0pi+
# patterns=['protonup10','protondn10','protonup12','protondn12'] # for physical ensemble NJN old 
# patterns = ['dt10_protonup','dt10_protondn','dt12_protonup','dt12_protondn','dt14_protonup','dt14_protondn'] # for NJN
# patterns = ['N_IL','N_pi0Loop','N_sigmaLoop','T_IL','N_pi0Loop_IL','N_sigmaLoop_IL'] # for QuarkLoops_pi0_insertion
# patterns = ['N_pi0Insert','N_sigmaInsert'] # for pi0Insertion


projList={}
projList['N']=[
    ((0,0,0),'a'),\
    ((0,0,1),'a'),((0,0,-1),'a'),((1,0,0),'a'),\
    ((0,1,1),'a'),\
    ((1,1,1),'a')
]
projList['Npi']=[
    ((0,0,0),('N1\\pi1','a')),((0,0,0),('N2\\pi2','a')),((0,0,0),('N3\\pi3','a')),\
    ((0,0,1),('N1\\pi0','a')),((0,0,1),('N0\\pi1','a')),((0,0,1),('N2\\pi1','a')),((0,0,1),('N2\\pi1','b')),\
        ((0,0,-1),('N1\\pi0','a')),((1,0,0),('N1\\pi0','a')),\
    ((0,1,1),('N2\\pi0','a')),((0,1,1),('N1\\pi1','a')),((0,1,1),('N1\\pi1','b')),((0,1,1),('N0\\pi2','a')),\
    ((1,1,1),('N3\\pi0','a')),((1,1,1),('N2\\pi1','a')),((1,1,1),('N2\\pi1','b')),((1,1,1),('N1\\pi2','a')),((1,1,1),('N1\\pi2','b')),((1,1,1),('N0\\pi3','a')),\

    # for N*pi
    ((0,0,0),('N0\\pi0','a')),\

    # for G1u
    ((0,0,0),('N0\\pi0','G1u_a'))
]

# This list includes all states containing an N*, whose operator is anti-Hermite conjugate in the source and sink.
# An additional sign will be implemented when the source has an N*.
NStarList=[
    ((0,0,0),('N0\\pi0','a')) 
] 

# main
coef_N=read_pickle(irrepCoefPathN)
coef_Npi=read_pickle(irrepCoefPathNpi)

tolerance=1e-16
irrepGDic={'2C2v':'G','2C4v':'G1','2Oh':'G1g','2C3v':'G'}

# proj=(p_tot,occ,lambda); occ=occ for N, occ=(mom2_Npi,occ) for Npi
projDic={}
projDic['N']={}
for group in coef_N.keys():
    for p_tot in coef_N[group].keys():
        irrep=irrepGDic[group]
        b_lab='b1'
        for lam in ['l1','l2']:
            for occ in coef_N[group][p_tot][irrep][b_lab][lam].keys():
                proj=(p_tot,occ,lam)
                projDic['N'][proj]=[]
                for a in range(4):
                    t=coef_N[group][p_tot][irrep][b_lab][lam][occ]['cg5'][a]
                    if np.abs(t) < tolerance:
                        continue
                    projDic['N'][proj].append((t,a))

        if group=='2Oh':
            irrep='G1u'
            b_lab='b1'
            for lam in ['l1','l2']:
                for occ in coef_N[group][p_tot][irrep][b_lab][lam].keys():
                    proj=(p_tot,'G1u_'+occ,lam)
                    projDic['N'][proj]=[]
                    for a in range(4):
                        t=coef_N[group][p_tot][irrep][b_lab][lam][occ]['cg5'][a]
                        if np.abs(t) < tolerance:
                            continue
                        projDic['N'][proj].append((t,a))

projDic['Npi']={}
for group in coef_Npi.keys():
    for p_tot in coef_Npi[group].keys():
        for mom2_Npi in coef_Npi[group][p_tot].keys():
            irrep=irrepGDic[group]
            b_lab='b1'
            for lam in ['l1','l2']:
                for occ in coef_Npi[group][p_tot][mom2_Npi][irrep][b_lab][lam].keys():
                    proj=(p_tot,(mom2_Npi,occ),lam)
                    projDic['Npi'][proj]=[]
                    for mom_Npi in coef_Npi[group][p_tot][mom2_Npi][irrep][b_lab][lam][occ].keys():
                        for a in range(4):
                            t = coef_Npi[group][p_tot][mom2_Npi][irrep][b_lab][lam][occ][mom_Npi]['cg5;g5'][a]
                            if np.abs(t) < tolerance:
                                continue
                            projDic['Npi'][proj].append((t,a,mom_Npi[1]))
            
            if group=='2Oh':
                irrep='G1u'
                b_lab='b1'
                for lam in ['l1','l2']:
                    for occ in coef_Npi[group][p_tot][mom2_Npi][irrep][b_lab][lam].keys():
                        proj=(p_tot,(mom2_Npi,'G1u_'+occ),lam)
                        projDic['Npi'][proj]=[]
                        for mom_Npi in coef_Npi[group][p_tot][mom2_Npi][irrep][b_lab][lam][occ].keys():
                            for a in range(4):
                                t = coef_Npi[group][p_tot][mom2_Npi][irrep][b_lab][lam][occ][mom_Npi]['cg5;g5'][a]
                                if np.abs(t) < tolerance:
                                    continue
                                projDic['Npi'][proj].append((t,a,mom_Npi[1]))

patDic={}
for pat in ['N_2pt']:
    patDic[pat]=('2pt','N','N')
for pat in ['T','N_pi0Loop']:
    patDic[pat]=('2pt','N','Npi')
for pat in ['M_correct_2pt','B_2pt','W_2pt','Z_2pt']:
    patDic[pat]=('2pt','Npi','Npi')
for pat in ['protonup10','protondn10','protonup12','protondn12',\
            'dt10_protonup','dt10_protondn','dt12_protonup','dt12_protondn','dt14_protonup','dt14_protondn',\
                'N_IL']:
    patDic[pat]=('3pt','N','N')
for pat in ['B','W','Z','N_pi0Insert','N_sigmaInsert','T_IL','N_pi0Loop_IL','N_sigmaLoop_IL']:
    patDic[pat]=('3pt','N','Npi')

compList=['N_IL','T_IL','N_pi0Loop_IL','N_sigmaLoop_IL'] # patterns in this list have data storing complex numbers directly, instead of an additional dimension for real and imag

# proj2Dic={
#     'P0': [(1,'l1','l1'),(1,'l2','l2')],
#     'Pz': [(1,'l1','l1'),(-1,'l2','l2')],
#     'Px': [(1,'l1','l2'),(1,'l2','l1')],
#     'Py': [(1j,'l1','l2'),(-1j,'l2','l1')], # note tr( [[0,-1j],[1j,0]] C ) = -1j*C_{1,0} + 1j * C_{0,1}
# }
proj2Dic={
    'P0': [(1,'l1','l1'),(1,'l2','l2')],
    'Pzt': [(1,'l1','l1'),(-1,'l2','l2')],
    'Pz': [(1,'l1','l1'),(-1,'l2','l2')],
    'Px': [(1,'l1','l2'),(1,'l2','l1')],
    'Py': [(1j,'l1','l2'),(-1j,'l2','l1')],
}


os.makedirs(outPath, exist_ok=True)

for pat in patterns:
    (npt,aCase,bCase)=patDic[pat]
    if pat in ['N_2pt']:
        with h5py.File(outPath+pat+'.h5','w') as fw:
            with h5py.File(inPath+pat+'.h5') as fr:
                # load momList
                cfg=list(fr.keys())[0]
                momList=fr[cfg+'/mvec']
                momDic={} # key: pf1
                for i in range(len(momList)):
                    ele=momList[i].astype(int)
                    momDic[tuple(ele)]=i
                # do proj
                for cfg in fr.keys():
                    tFr=fr[cfg]
                    for cont in tFr.keys():
                        if cont.endswith('mvec'):
                            continue
                        for proja in projList[aCase]:
                            for projb in projList[bCase]:
                                if proja[0] != projb[0]:
                                    continue
                                for proj2 in ['P0']:
                                    def t_get(eleA,eleB):
                                        return eleA[0]*np.conj(eleB[0])*tFr[cont][:,momDic[proja[0]],0,eleA[1]*4+eleB[1]]
                                    t=np.sum([[coef2*t_get(eleA,eleB) \
                                                for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                            for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                    if pat not in compList:
                                        t=t[:,0]+1j*t[:,1]
                                    if projb in NStarList:
                                        t=-t
                                    fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                    print('_'.join([pat,cfg])+': done')
        print()
        continue

    if pat in ['T','N_pi0Loop','N_sigmaLoop']:
        with h5py.File(outPath+pat+'.h5','w') as fw:
            with h5py.File(inPath+pat+'.h5') as fr:
                # load momList
                cfg=list(fr.keys())[0]
                momList=fr[cfg+'/12/pi2=0_0_0/mvec']
                momDic={} # key: pf1
                for i in range(len(momList)):
                    ele=momList[i,3:6].astype(int)
                    momDic[tuple(ele)]=i
                # do proj
                for cfg in fr.keys():
                    tFr=fr[cfg+'/12']
                    for cont in tFr['pi2=0_0_0'].keys():
                        if cont.endswith('mvec'):
                            continue
                        for proja in projList[aCase]:
                            for projb in projList[bCase]:
                                if proja[0] != projb[0]:
                                    continue
                                for proj2 in ['P0']:
                                    def t_get(eleA,eleB):
                                        tKey='pi2='+str(eleB[2][0])+'_'+str(eleB[2][1])+'_'+str(eleB[2][2])+'/'+cont
                                        return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[proja[0]],0,eleA[1]*4+eleB[1]]
                                    t=np.sum([[coef2*t_get(eleA,eleB)
                                                for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                            for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                    if pat not in compList:
                                        t=t[:,0]+1j*t[:,1]
                                    if projb in NStarList:
                                        t=-t
                                    fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                    print('_'.join([pat,cfg])+': done')
        print()
        continue

    if pat in ['B_2pt','W_2pt','Z_2pt','M_correct_2pt']:
        with h5py.File(outPath+pat+'.h5','w') as fw:
            with h5py.File(inPath+pat+'.h5') as fr:
                # load momList
                cfg=list(fr.keys())[0]
                momList=fr[cfg+'/12/pi2=0_0_0/mvec']
                momDic={} # key: (p_tot,pf2)
                for i in range(len(momList)):
                    (elePf1,elePf2)=(momList[i,3:6].astype(int),momList[i,6:9].astype(int))
                    elePtot=elePf1+elePf2
                    momDic[(tuple(elePtot),tuple(elePf2))]=i
                # do proj
                for cfg in fr.keys():
                    tFr=fr[cfg+'/12']
                    for cont in tFr['pi2=0_0_0'].keys():
                        if cont.endswith('mvec'):
                            continue
                        for proja in projList[aCase]:
                            for projb in projList[bCase]:
                                if proja[0] != projb[0]:
                                    continue
                                for proj2 in ['P0']:
                                    def t_get(eleA,eleB):
                                        tKey='pi2='+str(eleB[2][0])+'_'+str(eleB[2][1])+'_'+str(eleB[2][2])+'/'+cont
                                        return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[(proja[0],eleA[2])],0,eleA[1]*4+eleB[1]]
                                    t=np.sum([[coef2*t_get(eleA,eleB)
                                                for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                            for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                    if pat not in compList:
                                        t=t[:,0]+1j*t[:,1]
                                    if projb in NStarList:
                                        t=-t
                                    fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                    print('_'.join([pat,cfg])+': done')
        print()
        continue

    if pat in ['protonup10','protondn10','protonup12','protondn12']:
        with h5py.File(outPath+pat+'.h5','w') as fw:
            with h5py.File(inPath+pat+'.h5') as fr:
                # load momList
                cfg=list(fr.keys())[0]
                momList=fr[cfg+'/mvec']
                momDic={} # key: (pc,pf1)
                for i in range(len(momList)):
                    elePf1=momList[i,3:6].astype(int)
                    elePc=momList[i,6:9].astype(int)
                    momDic[(tuple(elePc),tuple(elePf1))]=i
                # do proj
                for cfg in fr.keys():
                    tFr=fr[cfg]
                    for cont in tFr.keys():
                        if cont.endswith('mvec'):
                            continue
                        for proja in projList[aCase]:
                            for projb in projList[bCase]:
                                if proja[0] != (0,0,0):
                                    continue
                                pc=(projb[0][0]-proja[0][0],projb[0][1]-proja[0][1],projb[0][2]-proja[0][2])
                                if projb[0] == (0,0,0):
                                    proj2List=['P0','Px','Py','Pz']
                                else:
                                    proj2List=['P0','Pzt']
                                for proj2 in proj2List:
                                    def t_get(eleA,eleB):
                                        return eleA[0]*np.conj(eleB[0])*tFr[cont][:,momDic[pc,proja[0]],:,eleA[1]*4+eleB[1]]
                                    t=np.sum([[coef2*t_get(eleA,eleB)
                                                for eleA in projDic[aCase][(projb[0],proja[1],la)] \
                                                for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                                # In the irrepCoef files, when p_tot!=0, 'l1' means spin up in p_tot direction. We label it as zt (t means tilde).
                                                # Here projb[0] is also used in sink Nucleon case to make sure, 'l1' is spin up in zt-direction 
                                            for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                    if pat not in compList:
                                        t=t[:,:,0]+1j*t[:,:,1]
                                    if projb in NStarList:
                                        t=-t
                                    fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                    print('_'.join([pat,cfg])+': done')
        print()
        continue

    if pat in ['dt10_protonup','dt10_protondn','dt12_protonup','dt12_protondn','dt14_protonup','dt14_protondn',\
               'N_IL']:
        with h5py.File(outPath+pat+'.h5','w') as fw:
            with h5py.File(inPath+pat+'.h5') as fr:
                # load momList
                cfg=list(fr.keys())[0]
                momList=fr[cfg+'/mvec']
                momDic={} # key: (pc,pf1)
                for i in range(len(momList)):
                    elePf1=momList[i,3:6].astype(int)
                    elePc=momList[i,9:12].astype(int)
                    momDic[(tuple(elePc),tuple(elePf1))]=i
                # do proj
                for cfg in fr.keys():
                    tFr=fr[cfg]
                    for cont in tFr.keys():
                        if cont.endswith('mvec'):
                            continue
                        for proja in projList[aCase]:
                            for projb in projList[bCase]:
                                if proja[0] != (0,0,0):
                                    continue
                                pc=(projb[0][0]-proja[0][0],projb[0][1]-proja[0][1],projb[0][2]-proja[0][2])
                                if projb[0] == (0,0,0):
                                    proj2List=['P0','Px','Py','Pz']
                                else:
                                    proj2List=['P0','Pzt']
                                for proj2 in proj2List:
                                    def t_get(eleA,eleB):
                                        return eleA[0]*np.conj(eleB[0])*tFr[cont][:,momDic[pc,proja[0]],:,eleA[1]*4+eleB[1]]
                                    t=np.sum([[coef2*t_get(eleA,eleB)
                                                for eleA in projDic[aCase][(projb[0],proja[1],la)] \
                                                for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                                # In the irrepCoef files, when p_tot!=0, 'l1' means spin up in p_tot direction. We label it as zt (t means tilde).
                                                # Here projb[0] is also used in sink Nucleon case to make sure, 'l1' is spin up in zt-direction 
                                            for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                    if pat not in compList:
                                        t=t[:,:,0]+1j*t[:,:,1]
                                    if projb in NStarList:
                                        t=-t
                                    fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                    print('_'.join([pat,cfg])+': done')
        print()
        continue

    if pat in ['B','W','Z','N_pi0Insert','N_sigmaInsert','T_IL','N_pi0Loop_IL','N_sigmaLoop_IL']:
        with h5py.File(outPath+pat+'.h5','w') as fw:
            with h5py.File(inPath+pat+'.h5') as fr:
                # load momList
                cfg=list(fr.keys())[0]
                momList=fr[cfg+'/12/pi2=0_0_0/mvec']
                momDic={} # key: pc
                for i in range(len(momList)):
                    ele=momList[i,9:12].astype(int)
                    momDic[tuple(ele)]=i
                # do proj
                for cfg in fr.keys():
                    tFr=fr[cfg+'/12']
                    for cont in tFr['pi2=0_0_0'].keys():
                        if cont.endswith('mvec'):
                            continue
                        for proja in projList[aCase]:
                            for projb in projList[bCase]:
                                if proja[0] != (0,0,0):
                                    continue
                                pc=(projb[0][0]-proja[0][0],projb[0][1]-proja[0][1],projb[0][2]-proja[0][2])
                                if projb[0] == (0,0,0):
                                    proj2List=['P0','Px','Py','Pz']
                                else:
                                    proj2List=['P0','Pzt']
                                for proj2 in proj2List:
                                    def t_get(eleA,eleB):
                                        tKey='pi2='+str(eleB[2][0])+'_'+str(eleB[2][1])+'_'+str(eleB[2][2])+'/'+cont
                                        return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[pc],:,eleA[1]*4+eleB[1]]
                                    t=np.sum([[coef2*t_get(eleA,eleB)
                                                for eleA in projDic[aCase][(projb[0],proja[1],la)] \
                                                for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                                # In the irrepCoef files, when p_tot!=0, 'l1' means spin up in p_tot direction. We label it as zt (t means tilde).
                                                # Here projb[0] is also used in sink Nucleon case to make sure, 'l1' is spin up in zt-direction 
                                            for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                    if pat not in compList:
                                        t=t[:,:,0]+1j*t[:,:,1]
                                    if projb in NStarList:
                                        t=-t
                                    fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                    print('_'.join([pat,cfg])+': done')
        print()
        continue

print('Done!')