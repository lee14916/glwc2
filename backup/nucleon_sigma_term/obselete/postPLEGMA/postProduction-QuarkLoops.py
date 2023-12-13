'''
nohup python3 -u postProduction-QuarkLoops.py > log/postProduction-QuarkLoops.out &

datastructure before:
(src)/mvec
(src)/stoc(seed)_(Nstoc)/(up or dn) # make sure only one of up and dn exists

after: (vacuum NOT subtracted)
(cfg)/mvec
(cfg)/pi0Loop: for i*(ubar g5 u - dbar g5 d)/sqrt(2)
(cfg)/u+d
(cfg)/u-d
... also datasets for the corresponding vacuum expectation values.

Will also be files for unmerged results in pat/(cfg).h5.
'''

import h5py,os
import numpy as np

# input
inPath = './output_data/'
outPath = './post_data/'
outPathMerged= './post_data-merged/'
patterns=['pi0Loop','insertLoop','sigmaLoop']
# patterns=['sigmaLoop']
namePre = 'Diagram'; namePost = '.h5'

def getFilePath(cfg,pat):
    return inPath+cfg+'/'+namePre+cfg+'_'+pat+namePost
def getFilePathOut(cfg,pat):
    return outPath+cfg+'/'+namePre+cfg+'_'+pat+namePost

templateCfgIndex=4

# cfgList
cfgList=[]
for cfg in os.listdir(inPath):
    if not (cfg[:4].isdecimal() and (len(cfg) in [4,5])):
        continue
    cfgList.append(cfg)
cfgList.sort()
Ncfg=len(cfgList)

# conjClass
conjClass=['id','g5','g5g1','g5g2','g5g3','g5g4']
gListDic={
    'pi0Loop':['g5'],
    'sigmaLoop':['id'],
    'insertLoop':['id','g1','g2','g3','g4','g5','g5g1','g5g2','g5g3','g5g4']
}


def complexConverting(t):
    assert(t.shape[-1]==2)
    return t[...,0]+1j*t[...,1]

# main
for pat in patterns:
    data={}
    gList=gListDic[pat]

    cfg=cfgList[templateCfgIndex]
    with h5py.File(getFilePath(cfg,pat)) as f:
        src='sx00sy00sz00st00'
        momList=f[src]['mvec'][()]
        momDic={}
        for i in range(len(momList)):
            momDic[tuple(momList[i])]=i

        momListNega=-momList
        momNegaIndexList=[momDic[tuple(momListNega[i])] for i in range(len(momList))]

    for cfg in cfgList:
        # if cfg == '0024':
        #     break

        try:
            with h5py.File(getFilePath(cfg,pat)) as f:
                f.keys()
        except:
            print('_'.join([pat,cfg]) + ': Fail to open')
            continue

        data[cfg]=0; Nstoc=0
        with h5py.File(getFilePath(cfg,pat)) as f:
            src='sx00sy00sz00st00'
            for stoc in f[src].keys():
                if not stoc.startswith('stoc'):
                    continue
                flas=f[src][stoc].keys()
                assert(len(flas)==1)
                fla=list(flas)[0]
                assert(fla in ['up','dn'])

                t_stoc=int(stoc.split('_')[1])
                Nstoc+=t_stoc
                t=complexConverting(f[src][stoc][fla])

                sgnConj=np.array([1 if gList[i] in conjClass else -1 for i in range(len(gList))])
                if fla=='up':
                    t_up=t
                    t_dn=np.conj(t[:,momNegaIndexList])*sgnConj[None,None,:]
                elif fla=='dn':
                    t_up=np.conj(t[:,momNegaIndexList])*sgnConj[None,None,:]
                    t_dn=t

                data[cfg] += t_stoc * np.array([t_up+t_dn,t_up-t_dn])

        if Nstoc !=0:
            data[cfg]=data[cfg]/Nstoc
        else:
            del data[cfg]

        print('_'.join([pat,cfg])+': Nstoc='+str(Nstoc))
    print()

    Ntc=len(data[cfgList[templateCfgIndex]][0])

    # This part is for subtracting the vacuum expectation values.
    # if 'g5' in gList:
    #     mom=momDic[(0,0,0)]
    #     gamma=gList.index('g5'); iso=1 # vec for ubar g5 u - dbar g5 d
    #     vev=np.mean([data[cfg][iso,:,mom,gamma] for cfg in data.keys()])
    #     vev=np.array([[[[vev if (i_iso==iso and i_mom==mom and i_gamma==gamma) else 0
    #            for i_gamma in range(len(gList))] for i_mom in range(len(momList))] for tc in range(Ntc)] for i_iso in range(2)])
    #     for cfg in data.keys():
    #         data[cfg]-=vev

    # if 'id' in gList:
    #     mom=momDic[(0,0,0)]
    #     gamma=gList.index('id'); iso=0 # vec for ubar u + dbar d
    #     vev=np.mean([data[cfg][iso,:,mom,gamma] for cfg in data.keys()])
    #     vev=np.array([[[[vev if (i_iso==iso and i_mom==mom and i_gamma==gamma) else 0
    #            for i_gamma in range(len(gList))] for i_mom in range(len(momList))] for tc in range(Ntc)] for i_iso in range(2)])
    #     for cfg in data.keys():
    #         data[cfg]-=vev

    os.makedirs(outPathMerged, exist_ok=True)
    with h5py.File(outPathMerged+pat+'.h5','w') as f:
        if pat == 'pi0Loop':
            for cfg in data.keys():
                f.create_dataset(cfg+'/mvec',data=momList)
                f.create_dataset(cfg+'/pi0Loop',data=data[cfg][1,:,:,:]*1j/np.sqrt(2)) # 1j for pion sign

                mom=momDic[(0,0,0)]; gamma=gList.index('g5'); iso=1
                vev=np.mean(data[cfg][iso,:,mom,gamma])
                f.create_dataset(cfg+'/pi0Loop_g5_vev',data=vev*1j/np.sqrt(2))

        elif pat == 'sigmaLoop':
            for cfg in data.keys():
                f.create_dataset(cfg+'/mvec',data=momList)
                f.create_dataset(cfg+'/sigmaLoop',data=data[cfg][0,:,:,:]*1/np.sqrt(2)) # 1j for pion sign

                mom=momDic[(0,0,0)]; gamma=gList.index('id'); iso=0
                vev=np.mean(data[cfg][iso,:,mom,gamma])
                f.create_dataset(cfg+'/sigmaLoop_id_vev',data=vev/np.sqrt(2))

        elif pat == 'insertLoop':
            for cfg in data.keys():
                f.create_dataset(cfg+'/mvec',data=momList)
                f.create_dataset(cfg+'/u+d',data=data[cfg][0,:,:,:])
                f.create_dataset(cfg+'/u-d',data=data[cfg][1,:,:,:])

                if 'g5' in gList:
                    mom=momDic[(0,0,0)]; gamma=gList.index('g5'); iso=1
                    vev=np.mean(data[cfg][iso,:,mom,gamma])
                    f.create_dataset(cfg+'/u-d_g5_vev',data=vev)
                if 'id' in gList:
                    mom=momDic[(0,0,0)]; gamma=gList.index('id'); iso=0
                    vev=np.mean(data[cfg][iso,:,mom,gamma])
                    f.create_dataset(cfg+'/u+d_id_vev',data=vev)
                    
        else:
            print(pat + ' not supported!')

    with h5py.File(outPathMerged+pat+'.h5') as fr:
        for cfg in cfgList:
            os.makedirs(outPath+cfg, exist_ok=True)
            with h5py.File(getFilePathOut(cfg,pat),'w') as fw:
                fw.copy(fr[cfg],fw)

print('Done!')