'''
nohup python3 -u postAvg-NJNpi.py > log/postAvg-NJNpi.out

sumFunc
@input dataset
@output 
    -1: remove
    0: copy
    String: rename
    Tuple (coef, dataset) : output dataset += coef * input dataset

'''
import os, shutil, h5py, re
import numpy as np

inPath='./avg-merge/'
outPath='./avg-post/'

def run(inFile, outFile, sumFunc):
    with h5py.File(inFile) as fr, h5py.File(outFile,'w') as fw:
        datasets=[]
        def visitor_func(name, node):
            if not isinstance(node, h5py.Dataset):
                return
            datasets.append(name)

        fr.visititems(visitor_func)

        sumDic={}
        for name in datasets:
            res = sumFunc(name)
            if res == 0:
                fw.copy(fr[name],fw,name=name)
            elif type(res) == str:
                fw.copy(fr[name],fw,name=res)
            elif type(res) == tuple:
                (coef,target) = res
                if target not in sumDic:
                    sumDic[target]=0
                sumDic[target] += coef * fr[name][()]
            else:
                raise Exception("not supported res: ",res)
            
        for target in sumDic:
            fw.create_dataset(target,data=sumDic[target])

os.makedirs(outPath)
for file in os.listdir(inPath):
    inFile=inPath+file
    outFile=outPath+file
    pat = file[:-3]

    if pat in ['D1ii','D1ii-pi0i']:
        tu=1/np.sqrt(2)
        td=-1/np.sqrt(2)
        def sumFunc(name):
            t=name.split('/')
            cont=t[-1]
            if cont in ['D1ii'+str(i) for i in [1,2,3,4]]:
                newCont='/'.join(t[:-1]+['p&pi0_p'])
                return (tu,newCont)
            elif cont in ['D1ii'+str(i) for i in [9,10]]:
                newCont='/'.join(t[:-1]+['p&pi0_p'])
                return (td,newCont)
            elif cont in ['D1ii'+str(i) for i in [13,14,15,16]]:
                newCont='/'.join(t[:-1]+['n&pi+_p'])
                return (1,newCont)
            return -1  
        run(inFile,outFile,sumFunc)
        print(pat,': done')
    else:
        os.rename(inFile,outFile)
        print(pat,': done')
print('Done!')