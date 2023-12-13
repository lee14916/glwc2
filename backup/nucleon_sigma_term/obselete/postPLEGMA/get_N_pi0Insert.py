'''
nohup python3 -u get_N_pi0Insert.py > log/get_N_pi0Insert.out &
'''

import os, re, h5py
import numpy as np

# input
totalT=48; maxSSS=16
totalL=totalT//2 # used when mulMomPhase
NPack=totalT//maxSSS
tfList=[10,12,14]
# pat='pi0Insert'
pat='sigmaInsert'

inDirPI='/p/project/pines/li47/code/scratch/run/nucleon_sigma_term/cA211.53.24/pi0Insertion/output_data/'
def getPIDir(cfg,st):
    return inDirPI+cfg+'/'+'Diagram'+cfg+'_'+st+'_'+pat+'.h5'

outDir='/p/project/pines/li47/code/scratch/run/nucleon_sigma_term/cA211.53.24/pi0Insertion/post_data/'

inDirN='/p/scratch/pines/fpittler/run/nucleon_sigma_term/cA211a.53.24/NJNpi/run/'
namePre = 'threept'; namePost = '.h5'
def getSrc(filename):
    return re.search('(sx[0-9]*sy[0-9]*sz[0-9]*st[0-9]*)',filename).group()
def getFiledir(cfg,src,pat):
    return inDirN+cfg+'/'+namePre+cfg+'_'+src+'_'+pat+namePost

templateCfgIndex=3
templateSrcIndex=0

# cfgSrcDic
cfgSrcDic={}
for cfg in os.listdir(inDirN):
    if not (cfg.isdecimal() and len(cfg)==4 ):
        continue
    cfgSrcDic[cfg]=[]
    for filename in os.listdir(inDirN+cfg):
        if not (filename.startswith(namePre) and filename.endswith(namePost)):
            continue
        src=getSrc(filename)
        cfgSrcDic[cfg].append(src)
    cfgSrcDic[cfg]=list(set(cfgSrcDic[cfg]))
cfgList=list(cfgSrcDic.keys())
cfgList.sort()
NCfg=len(cfgList)

# momDic
cfg=cfgList[templateCfgIndex]; src=cfgSrcDic[cfg][templateSrcIndex]
with h5py.File(getFiledir(cfg,src,'N_2pt')) as f:
    momDicN={}
    tF=f[list(f.keys())[0]]['mvec']
    for i in range(len(tF)):
        mom=tF[i] # pi1=pf1 here, need additional phase at the end
        momDicN[tuple(mom)]=i

with h5py.File(getFiledir(cfg,src,'B')) as f:
    momDicB={}
    pi2List=list(f[list(f.keys())[0]]['12'].keys())
    tF=f[list(f.keys())[0]]['12/pi2=0_0_0/mvec']
    for i in range(len(tF)):
        mom=tF[i,9:12] # pc
        momDicB[tuple(mom)]=i
    momListB={}
    for pi2Str in pi2List:
        momListB[pi2Str]=f[list(f.keys())[0]]['12'][pi2Str]['mvec'][()]

with h5py.File(getPIDir(cfg,src[-5:])) as f:
    momDicPI={}
    tF=f[list(f.keys())[0]]['mvec']
    for i in range(len(tF)):
        mom=tF[i] # pc
        momDicPI[tuple(mom)]=i
    pcList=tF[()]

assert(momDicB == momDicPI)

flaList=['up','dn']

# main
for cfg in cfgList:
    if not os.path.exists(outDir+cfg):
        os.mkdir(outDir+cfg)
    for src in cfgSrcDic[cfg]:
        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
        stStr='st'+st
        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
        try:
            with h5py.File(getFiledir(cfg,src,'N_2pt')) as f:
                tF=f[list(f.keys())[0]]['NP']
            # with h5py.File(getPIDir(cfg,st)) as f:
            #     tF=f[list(f.keys())[0]].keys()
            #     print(tF)
        except:
            print('-'.join([cfg,src])+': fail')
            continue

        getPhase=lambda mom: np.exp(1j*(2*np.pi/totalL)*(mom@[sx,sy,sz]))

        stList=[(st+maxSSS*i)%totalT for i in range(NPack)]
        datPIPre={}
        for t_st in stList:
            datPIPre[t_st]={}
            with h5py.File(getPIDir(cfg,'st'+'{:0>3}'.format(t_st))) as f:
                tF=f[list(f.keys())[0]]
                for pi2Str in pi2List:
                    datPIPre[t_st][pi2Str]={}
                    for fla in flaList:
                        datPIPre[t_st][pi2Str][fla]=tF[pi2Str][fla][:maxSSS,:,:,0]+1j*tF[pi2Str][fla][:maxSSS,:,:,1]

        datPI={}
        for pi2Str in pi2List:
            datPI[pi2Str]={}
            for fla in flaList:
                t=np.array([datPIPre[t_st][pi2Str][fla] for t_st in stList])
                t=np.concatenate(t,axis=0)
                datPI[pi2Str][fla]=t


        with h5py.File(outDir+cfg+'/Diagram'+cfg+'_'+src+'_N_'+pat+'.h5','w') as fw:
            with h5py.File(getFiledir(cfg,src,'N_2pt')) as f:
                for pi2Str in pi2List:
                    fw.create_dataset(list(f.keys())[0]+'/12/'+pi2Str+'/mvec',data=momListB[pi2Str])
                tF=f[list(f.keys())[0]]['NP'][()]
                for tf in tfList:
                    tempList=[(tf+maxSSS*i)%totalT for i in range(NPack)]
                    stExpand=[i//maxSSS for i in range(totalT)]
                    datN=tF[tempList,momDicN[(0,0,0)],0,:,0]+1j*tF[tempList,momDicN[(0,0,0)],0,:,1] # Here pf1=(0,0,0)
                    datN=np.array([datN*getPhase(pc) for pc in pcList])
                    for pi2Str in pi2List:
                        pi2=pi2Str[4:].split('_')
                        pi2=np.array([int(pi2[0]),int(pi2[1]),int(pi2[2])])
                        datNpi2=datN*getPhase(-pi2)
                        datTemp=np.tile(datNpi2,(10,1,1,1))
                        datTemp=np.transpose(datTemp,(2,1,0,3))
                        datTemp=datTemp[stExpand]
                        dat={}
                        for fla in flaList:
                            datTemp2=datPI[pi2Str][fla]
                            datTemp2=np.tile(datTemp2,(16,1,1,1))
                            datTemp2=np.transpose(datTemp2,(1,2,3,0))
                            t=datTemp*datTemp2
                            t=np.array([np.real(t),np.imag(t)])
                            dat[fla]=np.transpose(t,(1,2,3,4,0))
                        if pat=='pi0Insert':
                            preFactor=(-1)*np.conj(1j/np.sqrt(2)) # adj-sign and source pi0 
                            fw.create_dataset(list(f.keys())[0]+'/12/'+pi2Str+'/NP_u+d_deltat_'+str(tf),data=preFactor*(dat['up']-dat['dn']))
                            fw.create_dataset(list(f.keys())[0]+'/12/'+pi2Str+'/NP_u-d_deltat_'+str(tf),data=preFactor*(dat['up']+dat['dn']))
                        elif pat=='sigmaInsert':
                            preFactor=np.conj(1/np.sqrt(2)) # source sigma 
                            fw.create_dataset(list(f.keys())[0]+'/12/'+pi2Str+'/NP_u+d_deltat_'+str(tf),data=preFactor*(dat['up']+dat['dn']))
                            fw.create_dataset(list(f.keys())[0]+'/12/'+pi2Str+'/NP_u-d_deltat_'+str(tf),data=preFactor*(dat['up']-dat['dn']))                         

        print('-'.join([cfg,src])+': done') 

print('Done!')