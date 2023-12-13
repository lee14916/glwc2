'''
For sigma, G1g and G1u are interchanged
'''

import os, h5py, click
from pandas import read_pickle
import numpy as np
from datetime import datetime

# input
 
inPath='./avg-merge/'
outPath='./proj/'
irrepCoefPath='./irrepCoef/'
irrepCoefPathN=irrepCoefPath+'NEWcoeffsN.pkl'
irrepCoefPathNpi=irrepCoefPath+'NEWcoeffsNpi.pkl'

# NJN
patterns = ['NJN']
patternsDic={
    'NJN':[['pi0i']],
}

# NJNpi
patterns=['N_2pt','T','D1ii','M_correct_2pt','B_2pt','W_2pt','Z_2pt','B','W','Z']
patternsDic={
    # 'N_2pt':[['pi0i'],['pi0f'],['pi0f','pi0i'],['j'],['j','pi0i'],['pi0f','j'],['j&pi0i'],['pi0f&j'],\
    #          ['j','sigmai'],['j&sigmai'],['sigmaf&j']],
    'N_2pt':[['pi0i'],['pi0f'],['pi0f','pi0i'],['j'],['j','pi0i'],['pi0f','j'],['j&pi0i'],\
                ['j','sigmai'],['j&sigmai']],
    'T':[['pi0f'],['j']],
    'D1ii':[['pi0i'],['j']],
} 

def getTemplatePath(pat):
    return './templateFiles/'+pat+'.h5'

templateMomListPath=getTemplatePath('templateMomList')

# main
@click.command()
@click.option('-p','--patstodo',default='ALL')
def run(patstodo):
    startTime=datetime.now()
    print('Begin: '+str(startTime))
    print()

    pats=[]
    for pat_base in patterns:
        pats.append(pat_base)
        if pat_base in patternsDic:
            for case in patternsDic[pat_base]:
                pat="-".join([pat_base]+case)
                pats.append(pat)

    def get_t_pat(pat): # template pattern
        pat=pat.replace('sigma','pi0')
        if pat in ['N_2pt']:
            return 'N_2pt' #  N-N
        elif pat in ['T',\
                    'N_2pt-pi0i']: 
            return 'T' # N-Npi
        elif pat in ['D1ii','N_2pt-pi0f']:
            return 'D1ii' # Npi-N ?
        elif pat in ['B_2pt','W_2pt','Z_2pt','M_correct_2pt',\
                    'N_2pt-pi0f-pi0i','T-pi0f','D1ii-pi0i']: 
            return 'B_2pt' # Npi-Npi
        
        elif pat in ['NJN',\
                        'N_2pt-j']:
            return 'NJN' # N-J-N
        elif pat in ['B','W','Z',\
                    'N_2pt-j-pi0i','N_2pt-j&pi0i','T-j','NJN-pi0i']:
            return 'B' # N-J-Npi
        elif pat in ['N_2pt-pi0f-j','N_2pt-pi0f&j','D1ii-j','NJN-pi0f']:
            return 'NpiJN' # Npi-J-N

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

    projList={}
    projList['N']=[
        ((0,0,0),'a'),\
        ((0,0,1),'a'),\
        # ((0,0,-1),'a'),\
        # ((1,0,0),'a')
    ]
    projList['Npi']=[
        ((0,0,0),('N1\\pi1','a')),((0,0,0),('N2\\pi2','a')),
        ((0,0,1),('N1\\pi0','a')),((0,0,1),('N0\\pi1','a')),((0,0,1),('N2\\pi1','a')),((0,0,1),('N2\\pi1','b')),\
        # ((0,0,-1),('N1\\pi0','a')),((0,0,-1),('N0\\pi1','a')),\
        # ((1,0,0),('N1\\pi0','a')),((1,0,0),('N0\\pi1','a')),\
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
    for pat in ['T']:
        patDic[pat]=('2pt','N','Npi')
    for pat in ['D1ii']:
        patDic[pat]=('2pt','Npi','N')
    for pat in ['B_2pt']:
        patDic[pat]=('2pt','Npi','Npi')
    for pat in ['NJN']:
        patDic[pat]=('3pt','N','N')
    for pat in ['B']:
        patDic[pat]=('3pt','N','Npi')
    for pat in ['NpiJN']:
        patDic[pat]=('3pt','Npi','N')

    # proj2Dic={
    #     'P0': [(1,'l1','l1'),(1,'l2','l2')],
    #     'Pzt': [(1,'l1','l1'),(-1,'l2','l2')],
    #     'Pz': [(1,'l1','l1'),(-1,'l2','l2')],
    #     'Px': [(1,'l1','l2'),(1,'l2','l1')],
    #     'Py': [(1j,'l1','l2'),(-1j,'l2','l1')],
    # }
    proj2Dic={
        'P0': [(1/2,'l1','l1'),(1/2,'l2','l2')],
        'P01': [(1,'l1','l1')],
        'P02': [(1,'l2','l2')],
        'Pzt': [(1/2,'l1','l1'),(-1/2,'l2','l2')],
        'Pz': [(1/2,'l1','l1'),(-1/2,'l2','l2')],
        'Px': [(1/2,'l1','l2'),(1/2,'l2','l1')], 
        'Py': [(1j/2,'l1','l2'),(-1j/2,'l2','l1')],
    }

    proj2s_2pt=['P0','P01','P02']
    proj2s_3pt_zero=['P0','P01','P02','Pz']
    proj2s_3pt_nonz=proj2s_3pt_zero


    if patstodo != 'ALL':
        pats=patstodo.split(',')
    print(pats)
    print()

    os.makedirs(outPath, exist_ok=True)
    for pat in pats:
        t_pat = get_t_pat(pat)
        (npt,aCase,bCase)=patDic[t_pat]

        with h5py.File(templateMomListPath) as f:
            momList=f[t_pat][()]

        if t_pat in ['N_2pt']:
            with h5py.File(outPath+pat+'.h5','w') as fw:
                with h5py.File(inPath+pat+'.h5') as fr:
                    # load momList
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
                                    for proj2 in proj2s_2pt:
                                        def t_get(eleA,eleB):
                                            return eleA[0]*np.conj(eleB[0])*tFr[cont][:,momDic[proja[0]],0,eleA[1]*4+eleB[1]]
                                        t=np.sum([coef2*t_get(eleA,eleB) \
                                                    for (coef2,la,lb) in proj2Dic[proj2] \
                                                    for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                    for eleB in projDic[bCase][(projb[0],projb[1],lb)]
                                                ],axis=0)
                                        if projb in NStarList:
                                            t=-t
                                        fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                        print('_'.join([pat,cfg])+': done')
            print()
            continue

        if t_pat in ['T']:
            with h5py.File(outPath+pat+'.h5','w') as fw:
                with h5py.File(inPath+pat+'.h5') as fr:
                    # load momList
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
                                    for proj2 in proj2s_2pt:
                                        def t_get(eleA,eleB):
                                            tKey='pi2='+str(eleB[2][0])+'_'+str(eleB[2][1])+'_'+str(eleB[2][2])+'/'+cont
                                            return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[proja[0]],0,eleA[1]*4+eleB[1]]
                                        t=np.sum([coef2*t_get(eleA,eleB) \
                                                    for (coef2,la,lb) in proj2Dic[proj2] \
                                                    for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                    for eleB in projDic[bCase][(projb[0],projb[1],lb)]
                                                ],axis=0)
                                        if projb in NStarList:
                                            t=-t
                                        fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                        print('_'.join([pat,cfg])+': done')
            print()
            continue

        if t_pat in ['D1ii']:
            with h5py.File(outPath+pat+'.h5','w') as fw:
                with h5py.File(inPath+pat+'.h5') as fr:
                    # load momList
                    momDic={} # key: (p_tot,pf2)
                    for i in range(len(momList)):
                        (elePf1,elePf2)=(momList[i,3:6].astype(int),momList[i,6:9].astype(int))
                        elePtot=elePf1+elePf2
                        momDic[(tuple(elePtot),tuple(elePf2))]=i
                    # do proj
                    for cfg in fr.keys():
                        tFr=fr[cfg+'/12']
                        pi2Str='pi2=0_0_0'
                        for cont in tFr[pi2Str].keys():
                            if cont.endswith('mvec'):
                                continue
                            for proja in projList[aCase]:
                                for projb in projList[bCase]:
                                    if proja[0] != projb[0]:
                                        continue
                                    for proj2 in proj2s_2pt:
                                        def t_get(eleA,eleB):
                                            tKey=pi2Str+'/'+cont
                                            return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[(projb[0],eleA[2])],0,eleA[1]*4+eleB[1]]
                                        t=np.sum([coef2*t_get(eleA,eleB) \
                                                    for (coef2,la,lb) in proj2Dic[proj2] \
                                                    for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                    for eleB in projDic[bCase][(projb[0],projb[1],lb)]
                                                ],axis=0)
                                        if projb in NStarList:
                                            t=-t
                                        fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                        print('_'.join([pat,cfg])+': done')
            print()
            continue

        if t_pat in ['B_2pt']:
            with h5py.File(outPath+pat+'.h5','w') as fw:
                with h5py.File(inPath+pat+'.h5') as fr:
                    # load momList
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
                                    for proj2 in proj2s_2pt:
                                        def t_get(eleA,eleB):
                                            tKey='pi2='+str(eleB[2][0])+'_'+str(eleB[2][1])+'_'+str(eleB[2][2])+'/'+cont
                                            return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[(proja[0],eleA[2])],0,eleA[1]*4+eleB[1]]
                                        t=np.sum([coef2*t_get(eleA,eleB) \
                                                    for (coef2,la,lb) in proj2Dic[proj2] \
                                                    for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                    for eleB in projDic[bCase][(projb[0],projb[1],lb)]
                                                ],axis=0)
                                        if projb in NStarList:
                                            t=-t
                                        fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                        print('_'.join([pat,cfg])+': done')
            print()
            continue

        if t_pat in ['NJN']:
            with h5py.File(outPath+pat+'.h5','w') as fw:
                with h5py.File(inPath+pat+'.h5') as fr:
                    # load momList
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
                                        proj2List=proj2s_3pt_zero
                                    else:
                                        proj2List=proj2s_3pt_nonz
                                    for proj2 in proj2List:
                                        def t_get(eleA,eleB):
                                            return eleA[0]*np.conj(eleB[0])*tFr[cont][:,momDic[pc,proja[0]],:,eleA[1]*4+eleB[1]]
                                        # t=np.sum([[coef2*t_get(eleA,eleB)
                                        #             for eleA in projDic[aCase][(projb[0],proja[1],la)] \
                                        #             for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                        #             # In the irrepCoef files, when p_tot!=0, 'l1' means spin up in p_tot direction. We label it as zt (t means tilde).
                                        #             # Here projb[0] is also used in sink Nucleon case to make sure, 'l1' is spin up in zt-direction 
                                        #         for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                        t=np.sum([coef2*t_get(eleA,eleB) \
                                                    for (coef2,la,lb) in proj2Dic[proj2] \
                                                    for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                    for eleB in projDic[bCase][(projb[0],projb[1],lb)]
                                                ],axis=0)
                                        if projb in NStarList:
                                            t=-t
                                        fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                        print('_'.join([pat,cfg])+': done')
            print()
            continue

        if t_pat in ['B']:
            with h5py.File(outPath+pat+'.h5','w') as fw:
                with h5py.File(inPath+pat+'.h5') as fr:
                    # load momList
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
                                        proj2List=proj2s_3pt_zero
                                    else:
                                        proj2List=proj2s_3pt_nonz
                                    for proj2 in proj2List:
                                        def t_get(eleA,eleB):
                                            tKey='pi2='+str(eleB[2][0])+'_'+str(eleB[2][1])+'_'+str(eleB[2][2])+'/'+cont
                                            return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[pc],:,eleA[1]*4+eleB[1]]
                                        # t=np.sum([[coef2*t_get(eleA,eleB)
                                        #             for eleA in projDic[aCase][(projb[0],proja[1],la)] \
                                        #             for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                        #             # In the irrepCoef files, when p_tot!=0, 'l1' means spin up in p_tot direction. We label it as zt (t means tilde).
                                        #             # Here projb[0] is also used in sink Nucleon case to make sure, 'l1' is spin up in zt-direction 
                                        #         for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                        t=np.sum([coef2*t_get(eleA,eleB) \
                                                    for (coef2,la,lb) in proj2Dic[proj2] \
                                                    for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                    for eleB in projDic[bCase][(projb[0],projb[1],lb)]
                                                ],axis=0)
                                        if projb in NStarList:
                                            t=-t
                                        fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                        print('_'.join([pat,cfg])+': done')
            print()
            continue

        if t_pat in ['NpiJN']:
            with h5py.File(outPath+pat+'.h5','w') as fw:
                with h5py.File(inPath+pat+'.h5') as fr:
                    # load momList
                    momDic={} # key: pc pf2
                    for i in range(len(momList)):
                        elePf2=momList[i,6:9].astype(int)
                        elePc=momList[i,9:12].astype(int)
                        momDic[(tuple(elePc),tuple(elePf2))]=i
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
                                        proj2List=proj2s_3pt_zero
                                    else:
                                        proj2List=proj2s_3pt_nonz
                                    for proj2 in proj2List:
                                        def t_get(eleA,eleB):
                                            tKey='pi2=0_0_0/'+cont
                                            return eleA[0]*np.conj(eleB[0])*tFr[tKey][:,momDic[pc,eleA[2]],:,eleA[1]*4+eleB[1]]
                                        # t=np.sum([[coef2*t_get(eleA,eleB)
                                        #             for eleA in projDic[aCase][(projb[0],proja[1],la)] \
                                        #             for eleB in projDic[bCase][(projb[0],projb[1],lb)]] 
                                        #             # In the irrepCoef files, when p_tot!=0, 'l1' means spin up in p_tot direction. We label it as zt (t means tilde).
                                        #             # Here projb[0] is also used in sink Nucleon case to make sure, 'l1' is spin up in zt-direction 
                                        #         for (coef2,la,lb) in proj2Dic[proj2]],axis=(0,1))
                                        t=np.sum([coef2*t_get(eleA,eleB) \
                                                    for (coef2,la,lb) in proj2Dic[proj2] \
                                                    for eleA in projDic[aCase][(proja[0],proja[1],la)] \
                                                    for eleB in projDic[bCase][(projb[0],projb[1],lb)]
                                                ],axis=0)
                                        if projb in NStarList:
                                            t=-t
                                        fw.create_dataset(str(proja)+'/'+str(projb)+'/'+proj2+'/'+cfg+'/'+cont,data=t)
                        print('_'.join([pat,cfg])+': done')
            print()
            continue

    endTime=datetime.now()
    print('Begin: '+str(startTime))
    print('End: '+str(endTime))
    print('Cost: '+str(endTime- startTime))

if __name__ == '__main__':
    run()