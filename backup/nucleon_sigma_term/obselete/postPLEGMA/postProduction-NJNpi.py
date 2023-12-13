import os, shutil, click, h5py, re
import numpy as np
from datetime import datetime

# input
flag_D1ii_pi2Phase = True; L_total=24 # e^{-i*pi2*x}
flag_D1ii_normalize = True # 1/12
flag_T_seq2526 = True # -1
flag_BWZ_globalPhase = True # -1j 

flag_complexConverting=True

flag_removeInFile=False

inPath='./data_out/'
outPath='./data_post/'

def cfgQ(cfg):
    return cfg[-4:].isdecimal() and (len(cfg) in [4,5])
def fileQ(file):
    return file.endswith('.h5')
def getPat(file):
    return re.search('sx[0-9]*sy[0-9]*sz[0-9]*st[0-9]*_([\s\S]*?).h5',file).group(1)

# pre run
t_D1ii_normalize=1/12 if flag_D1ii_normalize else 1
t_T_seq2526=-1 if flag_T_seq2526 else 1
t_BWZ_globalPhase=-1j if flag_BWZ_globalPhase else 1

tu=1/np.sqrt(2) # pi0 u
td=-1/np.sqrt(2) # pi0 d

sumDic={}
sumDic['N_2pt']={
    'p_p':[(1,'NP')],
    'n_n':[(1,'N0')]
}

t_factor=t_T_seq2526
sumDic['T']={
    'p_n&pi+':[(1,'Tseq'+str(i)) for i in [11,12,13,14]],
    'p_p&pi0':[(tu,'Tseq'+str(i)) for i in [21,22,23,24]]+[(td*t_factor,'Tseq'+str(i)) for i in [25,26]],
}

t_factor=t_D1ii_normalize
sumDic['D1ii']={
    'n&pi+_p':[(1*t_factor,'D1ii'+str(i)) for i in [13,14,15,16]],
    'p&pi0_p':[(tu*t_factor,'D1ii'+str(i)) for i in [1,2,3,4]]+[(td*t_factor,'D1ii'+str(i)) for i in [9,10]],
}

sumDic['B_2pt']={
    'n&pi+_n&pi+':[(1,'B'+str(i)) for i in [13,14,15,16]],
    'n&pi+_p&pi0':[(tu,'B'+str(i)) for i in [17,18,19,20]]+[(td,'B'+str(i)) for i in []],
    'p&pi0_n&pi+':[(tu,'B'+str(i)) for i in [9,10,11,12]]+[(td,'B'+str(i)) for i in []],
    'p&pi0_p&pi0':[(tu*tu,'B'+str(i)) for i in [3,4,5,6]]+[(tu*td,'B'+str(i)) for i in []]+\
        [(td*tu,'B'+str(i)) for i in []]+[(td*td,'B'+str(i)) for i in [7,8]],
}
sumDic['W_2pt']={
    'n&pi+_n&pi+':[(1,'W'+str(i)) for i in [25,26,27,28]],
    'n&pi+_p&pi0':[(tu,'W'+str(i)) for i in [29,30,31,32]]+[(td,'W'+str(i)) for i in [33,34,35,36]],
    'p&pi0_n&pi+':[(tu,'W'+str(i)) for i in [17,18,19,20]]+[(td,'W'+str(i)) for i in [21,22,23,24]],
    'p&pi0_p&pi0':[(tu*tu,'W'+str(i)) for i in [5,6,7,8]]+[(tu*td,'W'+str(i)) for i in [9,10,11,12]]+\
        [(td*tu,'W'+str(i)) for i in [13,14,15,16]]+[(td*td,'W'+str(i)) for i in []],
}
sumDic['Z_2pt']={
    'n&pi+_n&pi+':[(1,'Z'+str(i)) for i in [15,16]],
    'n&pi+_p&pi0':[(tu,'Z'+str(i)) for i in []]+[(td,'Z'+str(i)) for i in [17,18,19,20]],
    'p&pi0_n&pi+':[(tu,'Z'+str(i)) for i in []]+[(td,'Z'+str(i)) for i in [11,12,13,14]],
    'p&pi0_p&pi0':[(tu*tu,'Z'+str(i)) for i in [5,6,7,8]]+[(tu*td,'Z'+str(i)) for i in []]+\
        [(td*tu,'Z'+str(i)) for i in []]+[(td*td,'Z'+str(i)) for i in [9,10]],
}
sumDic['M_correct_2pt']={
    'p&pi+_p&pi+':[(1,'MNPPP')],
    'n&pi+_n&pi+':[(1,'MN0PP')],
    'p&pi0_p&pi0':[(tu*tu,'MNPP01')]+[(td*td,'MNPP02')],
}

t_factor=t_BWZ_globalPhase
sumDic['B']={
    'p_j+_n&pi+':[(1*t_factor,'B'+str(i)) for i in [9,10,11,12]]+[(1*t_factor,'B'+str(i)) for i in []],
    'p_j-_n&pi+':[(1*t_factor,'B'+str(i)) for i in [9,10,11,12]]+[(-1*t_factor,'B'+str(i)) for i in []],

    'p_j+_p&pi0':[(tu*t_factor,'B'+str(i)) for i in [3,4,5,6]]+[(td*t_factor,'B'+str(i)) for i in []]+\
        [(tu*t_factor,'B'+str(i)) for i in []]+[(td*t_factor,'B'+str(i)) for i in [7,8]],
    'p_j-_p&pi0':[(tu*t_factor,'B'+str(i)) for i in [3,4,5,6]]+[(td*t_factor,'B'+str(i)) for i in []]+\
        [(-tu*t_factor,'B'+str(i)) for i in []]+[(-td*t_factor,'B'+str(i)) for i in [7,8]],
}
sumDic['W']={
    'p_j+_n&pi+':[(1*t_factor,'W'+str(i)) for i in [17,18,19,20]]+[(1*t_factor,'W'+str(i)) for i in [21,22,23,24]],
    'p_j-_n&pi+':[(1*t_factor,'W'+str(i)) for i in [17,18,19,20]]+[(-1*t_factor,'W'+str(i)) for i in [21,22,23,24]],

    'p_j+_p&pi0':[(tu*t_factor,'W'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'W'+str(i)) for i in [9,10,11,12]]+\
        [(tu*t_factor,'W'+str(i)) for i in [13,14,15,16]]+[(td*t_factor,'W'+str(i)) for i in []],
    'p_j-_p&pi0':[(tu*t_factor,'W'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'W'+str(i)) for i in [9,10,11,12]]+\
        [(-tu*t_factor,'W'+str(i)) for i in [13,14,15,16]]+[(-td*t_factor,'W'+str(i)) for i in []],
}
sumDic['Z']={
    'p_j+_n&pi+':[(1*t_factor,'Z'+str(i)) for i in []]+[(1*t_factor,'Z'+str(i)) for i in [11,12,13,14]],
    'p_j-_n&pi+':[(1*t_factor,'Z'+str(i)) for i in []]+[(-1*t_factor,'Z'+str(i)) for i in [11,12,13,14]],

    'p_j+_p&pi0':[(tu*t_factor,'Z'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'Z'+str(i)) for i in []]+\
        [(tu*t_factor,'Z'+str(i)) for i in []]+[(td*t_factor,'Z'+str(i)) for i in [9,10]],
    'p_j-_p&pi0':[(tu*t_factor,'Z'+str(i)) for i in [5,6,7,8]]+[(td*t_factor,'Z'+str(i)) for i in []]+\
        [(-tu*t_factor,'Z'+str(i)) for i in []]+[(-td*t_factor,'Z'+str(i)) for i in [9,10]],
}

sumMap={}
for pat in sumDic.keys():
    sumMap[pat]={}
    for contNew in sumDic[pat]:
        for coef,cont in sumDic[pat][contNew]:
            if cont not in sumMap[pat].keys():
                sumMap[pat][cont]=[]
            sumMap[pat][cont].append((coef,contNew))

def complexConverting(t):
    if flag_complexConverting:
        assert(t.shape[-1]==2)
        return t[...,0]+1j*t[...,1]
    return t

def runAux(inFile, outFile, sumFunc):
    with h5py.File(inFile) as fr, h5py.File(outFile,'w') as fw:
        datasets=[]
        def visitor_func(name, node):
            if not isinstance(node, h5py.Dataset):
                return
            datasets.append(name)

        fr.visititems(visitor_func)

        sums={}
        for name in datasets:
            res = sumFunc(name)
            if type(res) == list:
                for coef,contNew in res:
                    if contNew not in sums:
                        sums[contNew]=0
                    sums[contNew] += coef * complexConverting(fr[name][()])
            elif type(res) == tuple:
                (coef,contNew) = res
                if contNew not in sums:
                    sums[contNew]=0
                sums[contNew] += coef * complexConverting(fr[name][()])
            elif type(res) == str:
                if res == 'd':
                    continue
                if res == 'c':
                    res = name
                fw.copy(fr[name],fw,name=res)
            else:
                raise Exception(res+" not supported")
        
        for contNew in sums:
            fw.create_dataset(contNew,data=sums[contNew].astype('complex64'))

@click.command()
@click.option('-d','--division',default=0)
@click.option('-i','--index',default=0)
def run(division,index):
    assert(1<=index<=division)
    assert(inPath.endswith('/'))
    assert(outPath.endswith('/'))

    startTime=datetime.now()
    print('Begin: '+str(startTime))

    # main
    cfgList=os.listdir(inPath)
    cfgList=[cfg for cfg in cfgList if cfgQ(cfg)]
    cfgList.sort()
    Ncfg=len(cfgList)

    NcfgList=[Ncfg // division + (1 if x < Ncfg % division else 0)  for x in range(division)]
    Ncfg=NcfgList[index-1]
    NcfgList=[0]+list(np.cumsum(NcfgList))
    cfgList=cfgList[NcfgList[index-1]:NcfgList[index]]
    print('Ncfg='+str(Ncfg)+': from '+cfgList[0]+' to '+cfgList[-1]+'\n')

    for cfg in cfgList:
        os.makedirs(outPath+cfg, exist_ok=True)
        for file in os.listdir(inPath+cfg):
            if not fileQ(file):
                continue
            inFile=inPath+cfg+'/'+file
            outFile=outPath+cfg+'/'+file
            pat=getPat(file)

            if pat in ['P_2pt','N','D1ff']:
                if flag_removeInFile:
                    os.rename(inFile,outFile)
                else:
                    shutil.copy2(inFile,outFile)
                continue
            
            if pat in ['N_2pt','T','B_2pt','W_2pt','Z_2pt','M_correct_2pt','B','W','Z']:
                t_sumMap=sumMap[pat]
                def sumFunc(name):
                    pc=name.rsplit('/',1)
                    pre=pc[0]; cont=pc[1]
                    if cont in ['mvec']:
                        return 'c'
                    
                    if pat in ['B','W','Z']:
                        cc=cont.split('_deltat_')
                        cont=cc[0]; contPost=cc[1]
                        res = [(coef,pre+'/'+contNew+'_deltat_'+contPost) for coef,contNew in t_sumMap[cont]]
                    else:
                        res = [(coef,pre+'/'+contNew) for coef,contNew in t_sumMap[cont]]
                    return res
                
            elif pat in ['D1ii']:
                t_sumMap=sumMap[pat]
                (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',file).groups()
                (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                def sumFunc(name):
                    pc=name.rsplit('/',1)
                    pre=pc[0]; cont=pc[1]
                    pi2Str=re.search('pi2=-?[0-9]*_-?[0-9]*_-?[0-9]*',pre).group()
                    if cont in ['mvec']:
                        if flag_D1ii_pi2Phase:
                            pre=pre.replace(pi2Str,'pi2=0_0_0')
                            nameNew=pre+'/'+cont
                            return nameNew
                        return 'c'

                    res=t_sumMap[cont]
                    if flag_D1ii_pi2Phase:
                        assert(pi2Str=='pi2=-1_-1_-1')
                        pi2=pi2Str[4:].split('_')
                        pi2=np.array([int(pi2[0]),int(pi2[1]),int(pi2[2])])
                        getPhase=lambda mom: np.exp(1j*(2*np.pi/L_total)*(np.array([sx,sy,sz])@mom))
                        t_D1ii_pi2Phase = getPhase(pi2)
                        pre=pre.replace(pi2Str,'pi2=0_0_0')
                        res=[(coef*t_D1ii_pi2Phase,pre+'/'+contNew) for coef,contNew in res]
                        return res
                    else:
                        assert(pi2Str=='pi2=0_0_0')
                        res = [(coef,pre+'/'+contNew) for coef,contNew in t_sumMap[cont]]
                        return res
                    
            else:
                raise Exception(pat + " not supported")

            try:
                runAux(inFile,outFile,sumFunc)
                if flag_removeInFile:
                    os.remove(inFile)
                print(outFile,': done')
            except Exception as e:
                print(outFile,': fail')
                print(e)
        print()

    endTime=datetime.now()
    print('Begin: '+str(startTime))
    print('End: '+str(endTime))
    print('Cost: '+str(endTime- startTime))

if __name__ == '__main__':
    run()