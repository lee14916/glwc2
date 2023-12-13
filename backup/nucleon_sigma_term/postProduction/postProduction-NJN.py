'''
nohup python3 -u postProduction-NJN.py > log/postProduction-NJN.out &
'''

import os, shutil, click, h5py, re
import numpy as np
from datetime import datetime

# input
inPath = '/scratch/snx3000/fpittler/run/nucleon_sigma_term/cA211a.53.24/NJN/'
outPath = './post_data/'
namePre = 'Diagram'; namePost = '.h5'
# namePre = 'threept'; namePost = '.h5'

def getSrc(filename):
    return re.search('(sx[0-9]*sy[0-9]*sz[0-9]*st[0-9]*)',filename).group()
def getFilePath(cfg,src,pat):
    return inPath+cfg+'/'+namePre+cfg+'_'+src+'_'+pat+namePost
def getFilePathOut(cfg,src,pat):
    return outPath+cfg+'/'+'Diagram'+cfg+'_'+src+'_'+pat+namePost

tfList=[10,12,14]

def complexConverting(t):
    assert(t.shape[-1]==2)
    return t[...,0]+1j*t[...,1]

def run(cfg,src):
    outFile=getFilePathOut(cfg,src,'NJN')
    if os.path.isfile(outFile):
        return '-'.join([cfg,src])+': pass'
    with h5py.File(outFile,'w') as fw:
        for tf in tfList:
            with h5py.File(getFilePath(cfg,src,'dt'+str(tf)+'_protonup')) as fu, h5py.File(getFilePath(cfg,src,'dt'+str(tf)+'_protondn')) as fd:
                srcKey=list(fu.keys())[0]
                if tf == tfList[0]:
                    fw.create_dataset(srcKey+'/mvec',data=fu[srcKey]['mvec'])
                fw.create_dataset(srcKey+'/p_p_j+_deltat_'+str(tf),data=complexConverting(fu[srcKey]['MprotonUp'][()]+fd[srcKey]['MprotonDn']))
                fw.create_dataset(srcKey+'/p_p_j-_deltat_'+str(tf),data=complexConverting(fu[srcKey]['MprotonUp'][()]-fd[srcKey]['MprotonDn']))
    return '-'.join([cfg,src])+': done'

# cfgSrcDic
cfgSrcDic={}
for cfg in os.listdir(inPath):
    if not (cfg.isdecimal() and len(cfg)==4 ):
        continue
    cfgSrcDic[cfg]=[]
    for filename in os.listdir(inPath+cfg):
        if not (filename.startswith(namePre) and filename.endswith(namePost)):
            continue
        src=getSrc(filename)
        cfgSrcDic[cfg].append(src)
    cfgSrcDic[cfg]=list(set(cfgSrcDic[cfg]))
cfgList=list(cfgSrcDic.keys())
cfgList.sort()
NCfg=len(cfgList)

for cfg in cfgList:
    os.makedirs(outPath+cfg, exist_ok=True)
    for src in cfgSrcDic[cfg]:
        try:
            print(run(cfg,src))
        except Exception as e:
            print('-'.join([cfg,src]),': fail')
            print(e)
    print()

print('Done!')