''' 
'''
import os, shutil, click, h5py, re
import numpy as np

def dataQ(dataset):
    return not dataset.endswith('mvec')

def run(inFile,outFile):
    if os.path.isfile(outFile):
        return outFile+': pass'
    with h5py.File(outFile,'w') as fw, h5py.File(inFile) as fr:
        def visitor_func(name, node):
            if not isinstance(node, h5py.Dataset):
                return
            if dataQ(name):
                assert(node.shape[-1]==2)
                fw.create_dataset(name,data=node[...,0]+1j*node[...,1])
            else:
                fw.copy(node,fw)
        fr.visititems(visitor_func)
    return outFile+': done'

inPath='./post_data/'
outPath='./post_data2/'

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
            print(run(inFile,outFile))
            # os.remove(inFile)
        except Exception as e:
            print(outFile,': fail')
            print(e)
    print()

print('Done!')