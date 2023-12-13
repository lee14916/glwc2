'''
Steps of use:
# Put avgSrc.py & avgSrc.sh in the same directory
# Modify the (# input) part of avgSrc.py accordingly
# Run command '. avgSrc.sh', check the 'nohup.out', check Npack, dataset info.
### Dataset will be determined by the file with cfg,src as templateCfgIndex, templateSrcIndex
### Be careful that dataset does not miss anything you want
# Run command '. avgSrc.sh --division n' with n the number of nohup jobs that will be sent simultaneously
### It will avg over all src and store the unpacked results in avg/pat/cfg.h5
### SrcList will also be stored, so we don't need to do the same src over and over again
### Everytime before overwriting the avg/pat/cfg.h5 file, it will create a avg/pat/cfg.h5_backup file in case the overwriting fails.
### Log will be in 'log/avgSrc*.out'
# Run command '. avgSrc.sh --merge' 
### It will merge the avg/pat/cfg.h5 though cfg, and store the results in avg-merge/pat.h5

Supported data structure:
# Filename determined by cfg, src, pat
# Files are orgnized according to cfg: inPath/${cfg}/*.h5
# First level has only one group (normally src), will be replaced with cfg in the avg/pat.h5 file
# First dim of dataset is the time extent. Npack should divide the time extent
'''

import os, shutil, click, h5py, re
import numpy as np
from datetime import datetime

@click.command()
@click.option('-d','--division',default=0)
@click.option('-i','--index',default=0)
@click.option('-p','--patstodo',default='ALL')
def run(division,index,patstodo):
    assert(-1<=index<=division)
    startTime=datetime.now()
    print('Begin: '+str(startTime))

    # input 
    maxNsrc=999 # This option doesn't work well.
    space_total=24; time_total=48; tfList=[10,12,14]

    discFlag=True

    # NJN
    time_pack=48; Npack=1
    inPath = '/p/project/pines/li47/code/scratch/run/nucleon_sigma_term/cA211.53.24/NJN/post_data/'
    patterns = ['NJN']
    namePre = 'Diagram'; namePost = '.h5'
    if discFlag:
        patternsDic={
            'NJN':[['pi0i']],
        }

    # NJNpi
    time_pack=16; Npack=3
    inPath = '/project/s1174/lyan/code/scratch/run/nucleon_sigma_term/cA211.53.24/NJNpi_GEVP/data_post/'
    namePre = 'Diagram'; namePost = '.h5'
    patterns=['P_2pt','N_2pt','T','D1ii','M_correct_2pt','B_2pt','W_2pt','Z_2pt','B','W','Z']
    if discFlag:
        patternsDic={
            # 'N_2pt':[['pi0i'],['pi0f'],['pi0f','pi0i'],['j'],['j','pi0i'],['pi0f','j'],['j&pi0i'],['pi0f&j'],\
            #          ['j','sigmai'],['j&sigmai'],['sigmaf&j']],
            'N_2pt':[['pi0i'],['pi0f'],['pi0f','pi0i'],['j'],['j','pi0i'],['pi0f','j'],['j&pi0i'],\
                     ['j','sigmai'],['j&sigmai']],
            'T':[['pi0f'],['j']],
            'D1ii':[['pi0i'],['j']],
        } 

    outPath = './'

    def getSrc(filename):
        return re.search('(sx[0-9]*sy[0-9]*sz[0-9]*st[0-9]*)',filename).group()
    def getfilePath(cfg,src,pat):
        return inPath+cfg+'/'+namePre+cfg+'_'+src+'_'+pat+namePost
    def dataQ(dataset):
        return not dataset.endswith('mvec')
    def getTemplatePath(pat):
        return './templateFiles/'+pat+'.h5'
    
    # input for disconnected diagrams:
    complexConvertingFlag=True
    def complexConverting(t):
        if complexConvertingFlag and t.shape[-1]==2:
            return t[...,0]+1j*t[...,1]
        return t

    if discFlag:
        templateMomListPath=getTemplatePath('templateMomList')

        quarkLoopPath='/project/s1174/lyan/code/projectData/nucleon_sigma_term/cA211.53.24/QuarkLoops_pi0_insertion/data_post/'
        def getPath_mesonLoop(meson,cfg):
            return quarkLoopPath+cfg+'/Diagram'+cfg+'_'+meson+'Loop.h5'
        def getPath_jLoop(cfg):
            return quarkLoopPath+cfg+'/Diagram'+cfg+'_insertLoop.h5'
        pi0InsertPath='/project/s1174/lyan/code/projectData/nucleon_sigma_term/cA211.53.24/pi0Insertion/data_post/'
        def getPath_jmeson(meson,cfg,stStr):
            return pi0InsertPath+cfg+'/'+'Diagram'+cfg+'_'+stStr+'_'+meson+'Insert.h5'
        
        patbase2pat={}
        pat2patbase={}
        for pat_base in patternsDic:
            for case in patternsDic[pat_base]:
                pat="-".join([pat_base]+case)
                patbase2pat[(pat_base,tuple(case))]=pat
                pat2patbase[pat]=(pat_base,tuple(case))

        for pat in patterns:
            patbase2pat[(pat,())]=pat
            pat2patbase[pat]=(pat,())

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

            
        def hasInsert(pat):
            t_pat = get_t_pat(pat)
            return t_pat in ['NJN','B','NpiJN','NJN']

    # post_input
    assert(time_total==time_pack*Npack)
    
    pats=[]
    for pat in patterns:
        pats.append(pat)
        if discFlag:
            if pat in patternsDic:
                for case in patternsDic[pat]:
                    pat_new=patbase2pat[(pat,tuple(case))]
                    pats.append(pat_new)

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

    # merge
    if index==-1:
        print('Ncfg='+str(Ncfg)+': from '+cfgList[0]+' to '+cfgList[-1]+'\n')
        print('Npack='+str(Npack))

        if patstodo != 'ALL':
            pats=patstodo.split(',')
        print(pats)
        print()

        NsrcDic={}
        for pat in pats:
            os.makedirs(outPath+'avg-merge/', exist_ok=True)
            writePath=outPath+'avg-merge/'+pat+'.h5'

            with h5py.File(writePath,'w') as fw:
                NsrcDic[pat]={}
                for cfg in cfgList:
                    # if cfg not in cfgListTest:
                    #     continue
                    readPath=outPath+'avg/'+pat+'/'+cfg+'.h5'
                    with h5py.File(readPath) as fr:
                        Nsrc=len(fr['srcList'])
                        NsrcDic[pat][cfg]=Nsrc
                        if Nsrc==0:
                            print('-'.join([cfg,pat])+': '+'Nsrc='+str(Nsrc))
                            continue
                        fw.copy(fr[cfg],fw)    
                        print('-'.join([cfg,pat])+': Nsrc='+str(Npack)+'*'+str(Nsrc))    
                print()

        if discFlag:
            print('Checking if Nsrc[pat] == Nsrc[pat_base]: begin')
            for pat_base in patternsDic:
                for case in patternsDic[pat_base]:
                    pat=patbase2pat[(pat_base,tuple(case))]
                    if pat not in pats:
                        continue
                    if pat_base not in NsrcDic:
                        NsrcDic[pat_base]={}
                        for cfg in cfgList:
                            readPath=outPath+'avg/'+pat_base+'/'+cfg+'.h5'
                            with h5py.File(readPath) as fr:
                                Nsrc=len(fr['srcList'])
                                NsrcDic[pat_base][cfg]=Nsrc
                    for cfg in cfgList:
                        if NsrcDic[pat][cfg] != NsrcDic[pat_base][cfg]:
                            print('-'.join([cfg,pat])+': Nsrc not same with Nsrc_base!')
            print('Checking if Nsrc[pat] == Nsrc[pat_base]: end')
            print()

        endTime=datetime.now()
        print('Begin: '+str(startTime))
        print('End: '+str(endTime))
        print('Cost: '+str(endTime- startTime))
        return

    # datasetDic
    datasetDic = {}
    for pat in patterns:
        filePath=getTemplatePath(pat)
        datasetDic[pat]=[]
        def visitor_func(name, node):
            if isinstance(node, h5py.Dataset):
                if dataQ(name):
                    datasetDic[pat].append(name)
                
        with h5py.File(filePath,'r') as f:
            f[list(f.keys())[0]].visititems(visitor_func)

    # for disc:
    if discFlag:
        # N_2pt: pf1 -> pf1
        # T: pi2 pf1 -> pf1
        # D1ii: pi2 pf1 pf2 -> pf1 pf2
        # B_2pt: pi2 pf1 pf2 -> pf1 pf2
        # NJN:  pi2 pf1 pf2 pc -> pf1 pf2 pc
        # B: pi2 pf1 pf2 pc -> pf1 pf2 pc
        # NpiJN: pi2 pf1 pf2 pc -> pf1 pf2 pc
        momListDic={}
        momDicDic={}
        with h5py.File(templateMomListPath) as f:
            for pat in f.keys():
                momListDic[pat]={}
                if pat in ['N_2pt']:
                    momListDic[pat]['full']=f[pat][:,:]
                    momListDic[pat]['pf1']=f[pat][:,:]
                elif pat in ['T']:
                    momListDic[pat]['full']=f[pat][:,3:]
                    momListDic[pat]['pf1']=f[pat][:,3:6]
                elif pat in ['D1ii','B_2pt']:
                    momListDic[pat]['full']=f[pat][:,3:]
                    momListDic[pat]['pf1']=f[pat][:,3:6]
                    momListDic[pat]['pf2']=f[pat][:,6:9]
                elif pat in ['NJN','B','NpiJN']:
                    momListDic[pat]['full']=f[pat][:,3:]
                    momListDic[pat]['pf1']=f[pat][:,3:6]
                    momListDic[pat]['pf2']=f[pat][:,6:9]
                    momListDic[pat]['pc']=f[pat][:,9:12]
                else:
                    momListDic[pat]['full']=f[pat][()]

                momDicDic[pat]={}
                for i in range(len(momListDic[pat]['full'])):
                    momDicDic[pat][tuple(momListDic[pat]['full'][i])]=i

        momMapDic={}
        for pat_base in patternsDic:
            for case in patternsDic[pat_base]:
                pat=patbase2pat[(pat_base,tuple(case))]
                t_pat_base=get_t_pat(pat_base)
                t_pat=get_t_pat(pat)
                momMapDic[pat]={}
                if t_pat_base in ['N_2pt']:
                    momMapDic[pat]['base']=[momDicDic[t_pat_base][tuple(mom)] for mom in momListDic[t_pat]['pf1']]
                elif t_pat_base in ['T']:
                    momMapDic[pat]['base']=[momDicDic[t_pat_base][tuple(mom)] for mom in momListDic[t_pat]['pf1']]
                elif t_pat_base in ['D1ii']:
                    momMapDic[pat]['base']=[momDicDic[t_pat_base][tuple(mom)] for mom in np.concatenate([momListDic[t_pat]['pf1'],momListDic[t_pat]['pf2']],axis=1)]
                elif t_pat_base in ['B_2pt']:
                    momMapDic[pat]['base']=[momDicDic[t_pat_base][tuple(mom)] for mom in np.concatenate([momListDic[t_pat]['pf1'],momListDic[t_pat]['pf2']],axis=1)]
                elif t_pat_base in ['NJN']:
                    momMapDic[pat]['base']=[momDicDic[t_pat_base][tuple(mom)] for mom in np.concatenate([momListDic[t_pat]['pf1'],momListDic[t_pat]['pf2'],momListDic[t_pat]['pc']],axis=1)]
                elif t_pat_base in ['B']:
                    momMapDic[pat]['base']=[momDicDic[t_pat_base][tuple(mom)] for mom in np.concatenate([momListDic[t_pat]['pf1'],momListDic[t_pat]['pf2'],momListDic[t_pat]['pc']],axis=1)]

        filePath=getTemplatePath('T')
        with h5py.File(filePath) as f:
            pi2StrList=list(f[list(f.keys())[0]]['12'].keys())

        datasetMap={}
        for pat_base in patternsDic:
            # contList
            if pat_base in ['N_2pt','NJN']:
                filePath=getTemplatePath(pat_base)
                with h5py.File(filePath) as f:
                    contList=[]
                    for key in f[list(f.keys())[0]].keys():
                        if not key.endswith('mvec'):
                            if key.endswith('n_n'):
                                continue
                            contList.append(key)
            if pat_base in ['T']:
                filePath=getTemplatePath(pat_base)
                with h5py.File(filePath) as f: 
                    contList=[]
                    for key in f[list(f.keys())[0]]['12']['pi2=0_0_0'].keys():
                        if not key.endswith('mvec'):
                            contList.append(key)
            if pat_base in ['D1ii']:
                filePath=getTemplatePath(pat_base)
                with h5py.File(filePath) as f: 
                    contList=[]
                    assert(len(f[list(f.keys())[0]]['12'].keys())==1)
                    D1iiPi2Str=list(f[list(f.keys())[0]]['12'].keys())[0]
                    for key in f[list(f.keys())[0]]['12'][D1iiPi2Str].keys():
                        if not key.endswith('mvec'):
                            contList.append(key)   

            # datasetDic & datasetMap
            for case in patternsDic[pat_base]:
                pat=patbase2pat[(pat_base,tuple(case))]
                t_pat=get_t_pat(pat)
                datasetDic[pat]=[]
                datasetMap[pat]={}
                
                if t_pat in ['T']:
                    for pi2Str in pi2StrList:
                        for cont in contList:
                            dataset='12/'+pi2Str+'/'+cont
                            datasetDic[pat].append(dataset)
                            if pat_base in ['N_2pt']:
                                datasetMap[pat][dataset]=cont
                elif t_pat in ['D1ii']:
                    pi2Str = 'pi2=0_0_0'
                    for cont in contList:
                        dataset='12/'+pi2Str+'/'+cont
                        datasetDic[pat].append(dataset)
                        if pat_base in ['N_2pt']:
                            datasetMap[pat][dataset]=cont
                elif t_pat in ['B_2pt']:
                    for pi2Str in pi2StrList:
                        for cont in contList:
                            dataset='12/'+pi2Str+'/'+cont
                            datasetDic[pat].append(dataset)
                            if pat_base in ['N_2pt']:
                                datasetMap[pat][dataset]=cont
                            elif pat_base in ['T']:
                                datasetMap[pat][dataset]='12/'+pi2Str+'/'+cont
                            elif pat_base in ['D1ii']:
                                datasetMap[pat][dataset]='12/'+D1iiPi2Str+'/'+cont
                elif t_pat in ['NJN']:
                    for cont in contList:
                        for tf in tfList:
                            for iso in ['+','-']:
                                dataset=cont+'_j'+iso+'_deltat_'+str(tf)
                                datasetDic[pat].append(dataset)
                                if pat_base in ['N_2pt']:
                                    datasetMap[pat][dataset]=cont
                elif t_pat in ['B']:
                    if hasInsert(pat_base):
                        for pi2Str in pi2StrList:
                            for cont in contList:
                                dataset='12/'+pi2Str+'/'+cont
                                datasetDic[pat].append(dataset)
                                if pat_base in ['NJN']:
                                    datasetMap[pat][dataset]=cont
                    else:
                        for pi2Str in pi2StrList:
                            for cont in contList:
                                for tf in tfList:
                                    for iso in ['+','-']:
                                        dataset='12/'+pi2Str+'/'+cont+'_j'+iso+'_deltat_'+str(tf)
                                        datasetDic[pat].append(dataset)
                                        if pat_base in ['N_2pt']:
                                            datasetMap[pat][dataset]=cont
                                        elif pat_base in ['T']:
                                            datasetMap[pat][dataset]='12/'+pi2Str+'/'+cont
                elif t_pat in ['NpiJN']:
                    if hasInsert(pat_base):
                        for pi2Str in ['pi2=0_0_0']:
                            for cont in contList:
                                dataset='12/'+pi2Str+'/'+cont
                                datasetDic[pat].append(dataset)
                                if pat_base in ['NJN']:
                                    datasetMap[pat][dataset]=cont
                    else:
                        for pi2Str in ['pi2=0_0_0']:
                            for cont in contList:
                                for tf in tfList:
                                    for iso in ['+','-']:
                                        dataset='12/'+pi2Str+'/'+cont+'_j'+iso+'_deltat_'+str(tf)
                                        datasetDic[pat].append(dataset)
                                        if pat_base in ['N_2pt']:
                                            datasetMap[pat][dataset]=cont
                                        elif pat_base in ['T']:
                                            datasetMap[pat][dataset]='12/'+pi2Str+'/'+cont
                                        elif pat_base in ['D1ii']:
                                            datasetMap[pat][dataset]='12/'+D1iiPi2Str+'/'+cont
    cfgListTest=['0160','0168']
    
    # display dataset
    if index==0 or index==1:
        print('Ncfg='+str(Ncfg)+': from '+cfgList[0]+' to '+cfgList[-1]+'\n')
        for pat in patterns:
            print('Example: '+pat)
            for dataset in datasetDic[pat]:
                print(dataset)
            print()

    if index==0:
        return
    
    # avg
    NcfgList=[Ncfg // division + (1 if x < Ncfg % division else 0)  for x in range(division)]
    Ncfg=NcfgList[index-1]
    NcfgList=[0]+list(np.cumsum(NcfgList))
    cfgList=cfgList[NcfgList[index-1]:NcfgList[index]]
    print('Ncfg='+str(Ncfg)+': from '+cfgList[0]+' to '+cfgList[-1]+'\n')

    for cfg in cfgList:
        # if cfg not in cfgListTest:
        #     continue
        for pat in patterns:
            writePath=outPath+'avg/'+pat+'/'+cfg+'.h5'
            # init
            if not os.path.isfile(writePath):
                os.makedirs(outPath+'avg/'+pat, exist_ok=True)
                with h5py.File(writePath,'w') as fw:
                    fw.create_dataset('srcList',data=[])
                    for dataset in datasetDic[pat]:
                        fw.create_dataset(cfg+'/'+dataset,data=0)
            
            # read
            sumData={}
            with h5py.File(writePath) as fr:
                if len(fr['srcList'])==0:
                    srcList=[]
                else:
                    srcList=list(fr['srcList'].asstr()[()])
                flag = True
                NsrcPre = len(srcList)
                for dataset in datasetDic[pat]:
                    sumData[dataset]=fr[cfg+'/'+dataset][()]*NsrcPre

            if(maxNsrc <= NsrcPre):
                print('-'.join([cfg,pat]),': enough src')
                continue
                
            # sum data
            for src in cfgSrcDic[cfg]:
                if src in srcList:
                    print('-'.join([cfg,src,pat])+': pass')
                    continue
                
                # test if all datasets exist
                filePath=getfilePath(cfg,src,pat)
                try:
                    with h5py.File(filePath,'r') as fr:
                        ky=list(fr.keys())[0]
                        for dataset in datasetDic[pat]:
                            fr[ky+'/'+dataset]
                except Exception as e:
                    print('-'.join([cfg,src,pat])+': exception')
                    print(e)
                    continue
                
                # sum new data
                with h5py.File(filePath,'r') as fr:
                    ky=list(fr.keys())[0]
                    for dataset in datasetDic[pat]:
                        t=fr[ky+'/'+dataset][()]
                        shape=t.shape
                        if shape[0]%Npack!=0:
                            raise Exception('Npack cannot divide first index at '+dataset)
                        newShape=[Npack,shape[0]//Npack]+list(shape[1:])
                        t=np.reshape(t,newShape)
                        sumData[dataset] += complexConverting(np.mean(t,axis=0))
                srcList.append(src)
                print('-'.join([cfg,src,pat])+': done')

            # write data
            Nsrc=len(srcList)
            if Nsrc == NsrcPre:
                continue
            shutil.copy2(writePath,writePath+'_backup')
            with h5py.File(writePath,'w') as fw:
                fw.create_dataset('srcList',data=srcList)
                for dataset in datasetDic[pat]:
                    t=sumData[dataset]/Nsrc
                    fw.create_dataset(cfg+'/'+dataset,data=t.astype('complex64'))
            os.remove(writePath+'_backup')
        
        # for disc diagrams
        if discFlag:
            for pat_base in patternsDic:
                tempPath=outPath+'avg/'+pat_base+'/'+cfg+'.h5'
                with h5py.File(tempPath) as fr:
                    if len(fr['srcList'])==0:
                        srcList_base=[]
                    else:
                        srcList_base=list(fr['srcList'].asstr()[()])
                    Nsrc_base = len(srcList_base)

                for case in patternsDic[pat_base]:
                    pat=patbase2pat[(pat_base,tuple(case))]
                    t_pat_base=get_t_pat(pat_base)
                    t_pat=get_t_pat(pat)
                    writePath=outPath+'avg/'+pat+'/'+cfg+'.h5'
                    # init
                    if not os.path.isfile(writePath):
                        os.makedirs(outPath+'avg/'+pat, exist_ok=True)
                        with h5py.File(writePath,'w') as fw:
                            fw.create_dataset('srcList',data=[])
                            for dataset in datasetDic[pat]:
                                fw.create_dataset(cfg+'/'+dataset,data=0)

                    # read
                    sumData={}
                    with h5py.File(writePath) as fr:
                        if len(fr['srcList'])==0:
                            srcList=[]
                        else:
                            srcList=list(fr['srcList'].asstr()[()])
                        NsrcPre = len(srcList)
                        for dataset in datasetDic[pat]:
                            sumData[dataset]=fr[cfg+'/'+dataset][()]*NsrcPre

                    if(Nsrc_base == NsrcPre):
                        print('-'.join([cfg,pat]),': enough src')
                        continue

                    # sum data
                    for src in srcList_base:
                        if src in srcList:
                            print('-'.join([cfg,src,pat])+': pass')
                            continue
                        
                        # test if all datasets exist
                        filePath=getfilePath(cfg,src,pat_base)
                        try:
                            with h5py.File(filePath,'r') as fr:
                                ky=list(fr.keys())[0]
                                for dataset in datasetDic[pat_base]:
                                    fr[ky+'/'+dataset]
                        except Exception as e:
                            print('-'.join([cfg,src,pat])+': exception')
                            print(e)
                            continue

                        # sum new data
                        (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                        (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))

                        getPhase=lambda mom: np.exp(1j*(2*np.pi/space_total)*(np.array([sx,sy,sz])@mom))

                        with h5py.File(filePath,'r') as fr:
                            ky=list(fr.keys())[0]
                            for dataset in datasetDic[pat]:
                                dataset_base=datasetMap[pat][dataset]
                                t=fr[ky+'/'+dataset_base][()]
                                t=complexConverting(t) # time, mom, G_c, ab
                                t=t[:,momMapDic[pat]['base'],:,:]

                                if pat_base == 'NJN': # expand the time dimension to time_total
                                    assert(Npack==1)
                                    shape = t.shape
                                    shape0=shape[0]
                                    newShape = [time_total-shape0]+list(shape[1:])
                                    t=np.concatenate([t,np.zeros(newShape)],axis=0)

                                for t_meson in ['pi0','sigma']:
                                    if t_meson+'i' in case:
                                        pi2Str=re.search('pi2=-?[0-9]*_-?[0-9]*_-?[0-9]*',dataset).group()
                                        pi2=pi2Str[4:].split('_')
                                        pi2=np.array([int(pi2[0]),int(pi2[1]),int(pi2[2])])
                                        timeMap = [(st+i//time_pack*time_pack)%time_total for i in range(time_total)]
                                        with h5py.File(getPath_mesonLoop(t_meson,cfg)) as fl:
                                            t_loop=np.conj(fl[cfg][t_meson+'Loop'][:,momDicDic[t_meson+'Loop'][tuple(pi2)],0])
                                        t=t*t_loop[timeMap,None,None,None]*getPhase(-pi2)

                                for t_meson in ['pi0','sigma']:
                                    if t_meson+'f' in case:
                                        if hasInsert(pat_base):
                                            tf=dataset.split('_')[-1]
                                            tf=int(tf)
                                            timeMap = [(st+i//time_pack*time_pack+tf)%time_total for i in range(time_total)]
                                        else:
                                            timeMap = [(st+tf)%time_total for tf in range(time_total)]

                                        with h5py.File(getPath_mesonLoop(t_meson,cfg)) as fl:
                                            t_loop=fl[cfg][t_meson+'Loop'][:,:,0]

                                        momList=momListDic[t_pat]['pf2']
                                        momMap=[momDicDic[t_meson+'Loop'][tuple(mom)] for mom in momList]
                                        tPhase=np.array([getPhase(mom) for mom in momList])
                                        t_loop=t_loop[timeMap]

                                        t=t*t_loop[:,momMap,None,None]*tPhase[None,:,None,None]

                                if 'j' in case:
                                    tf=dataset.split('_')[-1]
                                    tf=int(tf)
                                    timeMap_base = [(tf+i//time_pack*time_pack)%time_total for i in range(time_total)]
                                    timeMap = [(st+tc)%time_total for tc in range(time_total)]

                                    with h5py.File(getPath_jLoop(cfg)) as fl:
                                        if '_j+_' in dataset:
                                            iso = 'u+d'
                                        else:
                                            assert('_j-_' in dataset)
                                            iso = 'u-d'
                                        t_loop=fl[cfg][iso][:,:,:]
                                    momList=momListDic[t_pat]['pc']
                                    momMap=[momDicDic['insertLoop'][tuple(mom)] for mom in momList]
                                    tPhase=np.array([getPhase(mom) for mom in momList])
                                    t_loop=t_loop[timeMap]
                                    t=t[timeMap_base,:,:,:]*t_loop[:,momMap,:,None]*tPhase[None,:,None,None]

                                for t_meson in ['pi0','sigma']:
                                    if 'j&'+t_meson+'i' in case:
                                        pi2Str=re.search('pi2=-?[0-9]*_-?[0-9]*_-?[0-9]*',dataset).group()
                                        pi2=pi2Str[4:].split('_')
                                        pi2=np.array([int(pi2[0]),int(pi2[1]),int(pi2[2])])
                                        tiList=[(st+time_pack*i)%time_total for i in range(Npack)]
                                        tPI={}
                                        for ti in tiList:
                                            with h5py.File(getPath_jmeson(t_meson,cfg,'st'+'{:0>3}'.format(ti))) as fl:
                                                if '_j+_' in dataset:
                                                    iso = 'j+'
                                                else:
                                                    assert('_j-_' in dataset)
                                                    iso = 'j-'
                                                tPI[ti]=fl[list(fl.keys())[0]][pi2Str][iso][:time_pack,:,:]
                                        ttPI=np.concatenate([tPI[ti] for ti in tiList],axis=0)

                                        tf=dataset.split('_')[-1]
                                        tf=int(tf)
                                        timeMap_base = [(tf+i//time_pack*time_pack)%time_total for i in range(time_total)]
                                        
                                        momList=momListDic[t_pat]['pc']
                                        momMap=[momDicDic[t_meson+'Insert'][tuple(mom)] for mom in momList]
                                        tPhase=np.array([getPhase(mom) for mom in momList])

                                        t=t[timeMap_base,:,:,:] * ttPI[:,momMap,:,None] * tPhase[None,:,None,None] * getPhase(-pi2)

                                for t_meson in ['pi0','sigma']:
                                    if t_meson+'f&j' in case:
                                        pass

                                if pat_base == 'NJN': # reduce the time dimension to time_total
                                    assert(Npack==1)
                                    t=t[:shape0]

                                shape=t.shape
                                if shape[0]%Npack!=0:
                                    raise Exception('Npack cannot divide first index at '+dataset)
                                newShape=[Npack,shape[0]//Npack]+list(shape[1:])
                                t=np.reshape(t,newShape)
                                sumData[dataset] += np.mean(t,axis=0)

                        srcList.append(src)
                        print('-'.join([cfg,src,pat])+': done')

                    # write data
                    Nsrc=len(srcList)
                    if Nsrc == NsrcPre:
                        continue
                    shutil.copy2(writePath,writePath+'_backup')
                    with h5py.File(writePath,'w') as fw:
                        fw.create_dataset('srcList',data=srcList)
                        for dataset in datasetDic[pat]:
                            t=sumData[dataset]/Nsrc
                            fw.create_dataset(cfg+'/'+dataset,data=t.astype('complex64'))
                    os.remove(writePath+'_backup')

        print()

    endTime=datetime.now()
    print('Begin: '+str(startTime))
    print('End: '+str(endTime))
    print('Cost: '+str(endTime- startTime))
 
if __name__ == '__main__':
    run()