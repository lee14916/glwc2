'''
nohup python3 -u get_Npi0Loop.py > log/get_Npi0Loop.out &
'''

import os, h5py, re
import numpy as np

# input
NPack = 48//16; maxSSS=16; totalT=48
totalL=totalT//2 # used when mulMomPhase

meson='pi0'
meson='sigma'

inPath='/p/project/pines/li47/code/scratch/fpittler/run/nucleon_sigma_term/cA211a.53.24/NJNpi/run/'
loopPath='./avg/'
outPath='/p/project/pines/li47/code/scratch/run/nucleon_sigma_term/cA211.53.24/QuarkLoops_pi0_insertion/post_data/'

patterns=['N_2pt']
namePre = 'threept'; namePost = '.h5'

def getSrc(filename):
    return re.search('(sx[0-9]*sy[0-9]*sz[0-9]*st[0-9]*)',filename).group()
def getInPath(cfg,src,pat):
    return inPath+cfg+'/'+namePre+cfg+'_'+src+'_'+pat+namePost
def getOutPath(cfg,src,pat):
    return outPath+cfg+'/'+'Diagram'+cfg+'_'+src+'_'+pat+namePost

templateCfgIndex=3
templateSrcIndex=0

# input: for template momList
patternsOutDic={'N_2pt':'N_'+meson+'Loop'}
# mvec3ptTemplate={}
deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)

outTemplatePath='/p/project/pines/li47/code/projectData/nucleon_sigma_term/cA211.53.24/NJNpi_N0pi+/sum/T/0000.h5'
with h5py.File(outTemplatePath) as f:
    pi2TemplateList=list(f['0000/12'].keys())
    outTemplateMomList=f['0000/12/pi2=0_0_0/mvec'][()] # pi2 pf1

# cfgSrcDic
cfgSrcDic={}
for cfg in os.listdir(inPath):
    if not (cfg[-4:].isdecimal() and (len(cfg) in [4,5])):
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
Ncfg=len(cfgList)

# momDic & momIndexList
cfg=cfgList[templateCfgIndex]; src=cfgSrcDic[cfg][templateSrcIndex]

momDicIn={} # pf1
with h5py.File(getInPath(cfg,src,patterns[0])) as f:
    t=deepKey(f,1)['mvec']
    for i in range(len(t)):
        momDicIn[tuple(t[i])]=i

momDicLoop={} # pi2
with h5py.File(loopPath+meson+'Loop.h5') as f:
    t=f[cfg]['mvec']
    for i in range(len(t)):
        momDicLoop[tuple(t[i])]=i

# main
print('NCfg='+str(Ncfg)+': from '+cfgList[0]+' to '+cfgList[-1]+'\n')

dataLoop={}
with h5py.File(loopPath+meson+'Loop.h5') as f:
    for cfg in cfgList:
        dataLoop[cfg]=f[cfg][meson+'Loop'][()]

for pat in patterns:
    for cfg in cfgList:
        # if cfg != '0012':
        #     continue
        if not os.path.exists(outPath+cfg):
            os.mkdir(outPath+cfg)
        for src in cfgSrcDic[cfg]:
            (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
            (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
            getPhase=lambda mom: np.exp(1j*(2*np.pi/totalL)*(mom@[sx,sy,sz]))
            try:
                with h5py.File(getInPath(cfg,src,pat)) as f:
                    tIn=deepKey(f,1)['NP'][:,:,:,:,0]+1j*deepKey(f,1)['NP'][:,:,:,:,1]
                    tMomIndList=[momDicIn[tuple(mom[3:6])] for mom in outTemplateMomList]
                    tIn=tIn[:,tMomIndList,:,:]

                with h5py.File(getOutPath(cfg,src,patternsOutDic[pat]),'w') as fw:
                    fw.create_dataset(src+'/12/pi2=0_0_0/mvec',data=outTemplateMomList)
                    for pi2Group in pi2TemplateList:
                        # if pi2Group != 'pi2=0_0_1':
                        #     continue
                        pi2=pi2Group[4:].split('_')
                        pi2=np.array([int(pi2[0]),int(pi2[1]),int(pi2[2])])
                        pi2Ind=momDicLoop[tuple(pi2)]
                        
                        tiList=[(st+i//maxSSS*maxSSS)%totalT for i in range(totalT)]

                        tLoop=np.conj(dataLoop[cfg][tiList,pi2Ind,0])
                        
                        tPhase=getPhase(-pi2)
                        tOut=tIn*tLoop[:,None,None,None]*tPhase 

                        tOut=np.array([np.real(tOut),np.imag(tOut)])
                        tOut=np.transpose(tOut,(1,2,3,4,0))
                        fw.create_dataset(src+'/12/'+pi2Group+'/NP_'+meson+'Loop',data=tOut) 
                        # print(tOut[:6,16,0,0])
                        # exit(0)
            
            except Exception as e:
                print('-'.join([pat,cfg,src])+': fail')
                print(e)
                continue
            print('-'.join([pat,cfg,src])+': done')
        print()

print('Done!')