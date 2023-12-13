'''
nohup python3 -u postProduction-pi0Insertion.py > log/postProduction-pi0Insertion.out &
'''

import os, shutil, click, h5py, re
import numpy as np

def complexConverting(t):
    assert(t.shape[-1]==2)
    return t[...,0]+1j*t[...,1]

def run(inFile,outFile,sumDic):
    if os.path.isfile(outFile):
        return outFile+': pass'
    with h5py.File(inFile) as fr, h5py.File(outFile,'w') as fw:
        for case in sumDic:
            t_group=sumDic[case][0][1]
            t_group_len=len(t_group)
            todoList=[]
            datasetDic={}

            def visitor_func(name, node):
                if not isinstance(node, h5py.Dataset):
                    return
                datasetDic[name]=0
                t=name.find(t_group)
                if t==-1:
                    return
                todoList.append((name[:t],name[t+t_group_len:]))
            fr.visititems(visitor_func)

            for ele in todoList:
                t_sum=0
                for (coe,key_sub) in sumDic[case]:
                    t_sum += coe * fr[ele[0]+key_sub+ele[1]][()]
                fw.create_dataset(ele[0]+case+ele[1],data=complexConverting(t_sum))

        holdList=[]
        def visitor_func(name, node):
            if not isinstance(node, h5py.Dataset):
                return
            for key in sumDic:
                for ele in sumDic[key]:
                    if ele[1] in name:
                        return
            holdList.append(name)
        fr.visititems(visitor_func)
        for ele in holdList:
            fw.copy(fr[ele],fw,name=ele)
    
    return outFile+': done'

#
sumDic={}
t=(-1)*np.conj(1j/np.sqrt(2)) # adj-sign and source pi0 
sumDic['pi0Insert']={
    'j+':[(t,'up'),(-t,'dn')],
    'j-':[(t,'up'),(t,'dn')],
}
t2=np.conj(1/np.sqrt(2)) # source sigma 
sumDic['sigmaInsert']={
    'j+':[(t2,'up'),(t2,'dn')],
    'j-':[(t2,'up'),(-t2,'dn')],
}

inPath='./data_out/'
outPath='./post_data/'

cfgList=[]
for cfg in os.listdir(inPath):
    if not (cfg[-4:].isdecimal() and (len(cfg) in [4,5])):
        continue
    cfgList.append(cfg)
cfgList.sort()

for cfg in cfgList:
    os.makedirs(outPath+cfg, exist_ok=True)
    for file in os.listdir(inPath+cfg):
        inFile=inPath+cfg+'/'+file
        outFile=outPath+cfg+'/'+file
        try:
            if file.endswith('pi0Insert.h5'):
                print(run(inFile,outFile,sumDic['pi0Insert']))
            else:
                assert(file.endswith('sigmaInsert.h5'))
                print(run(inFile,outFile,sumDic['sigmaInsert']))
            # os.remove(inFile)
        except Exception as e:
            print(outFile,': fail')
            print(e)
    print()

print('Done!')