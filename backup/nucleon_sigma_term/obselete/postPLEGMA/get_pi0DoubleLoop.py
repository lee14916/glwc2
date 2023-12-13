'''
nohup python3 -u get_pi0DoubleLoop.py > log/get_pi0DoubleLoop.out &

datastructure:
(cfg)/mvec # it is for pf2 (sink mom of pi0)
(cfg)/pi0DoubleLoop: dim=(total_t,N_mom)
'''

import h5py
import numpy as np

# input
inPath = './avg/'
outPath = inPath

# main
with h5py.File(inPath+'pi0Loop.h5') as fr:
    momList=fr[list(fr.keys())[0]]['mvec'][()]
    momDic={}
    for i in range(len(momList)):
        momDic[tuple(momList[i])]=i
    momListNega=-momList
    momNegaIndexList=[momDic[tuple(momListNega[i])] for i in range(len(momList))]

with h5py.File(outPath+'pi0DoubleLoop.h5','w') as fw:
    with h5py.File(inPath+'pi0Loop.h5') as fr:
        for cfg in fr.keys():
            fw.create_dataset(cfg+'/mvec',data=momList)
            tDatSink=fr[cfg]['pi0Loop'][:,:,0]
            tDatSrc=np.conj(tDatSink[:,momNegaIndexList])
            t=np.array([np.mean(np.roll(tDatSink,-tf,axis=0)*tDatSrc,axis=0) for tf in range(len(tDatSink))])
            fw.create_dataset(cfg+'/pi0DoubleLoop',data=t) # the conj already take all source signs into account

print('Done!')