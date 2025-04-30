'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u post2avgsrc.py -c @ > log/post2avgsrc.out & 
'''
import os, click, h5py, re
import numpy as np
from pandas import read_pickle
import auxx as aux

def sparseMatMul(sm,nda,axis):
    t=np.moveaxis(nda,axis,0)
    ts=t.shape
    t=np.reshape(t,(ts[0],-1))
    t=sm@t
    t=np.reshape(t,(-1,)+ts[1:])
    t=np.moveaxis(t,0,axis)
    return t
def fla2tf(fla):
    assert('_deltat_' in fla)
    return int(fla.split('_deltat_')[-1])
def src2ints(src):
    (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
    (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
    return (sx,sy,sz,st)
def get_phase(src_int,mom):
    (sx,sy,sz,st)=src_int
    return np.exp(1j*(2*np.pi/aux.lat_L)*(np.array([sx,sy,sz])@mom))

def mom_exchange_pi(mom):
    pi2,_,pf2,pc=list(mom[0:3]),list(mom[3:6]),list(mom[6:9]),list(mom[9:12])
    return list(-np.array(pf2))+[0,0,0]+list(-np.array(pi2))+pc
def moms_exchange_pi(moms):
    return [mom_exchange_pi(mom) for mom in moms]
    
@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'{aux.pathBaseTf}data_post/'+cfg+'/'
    outpath=f'{aux.pathBaseTf}data_avgsrc/'+cfg+'/'
    os.makedirs(outpath,exist_ok=True)
    
    with open(aux.path_diags_all) as f:
        diags=f.read().splitlines() 
    
    opabsDic=read_pickle(aux.path_opabsDic)
    auxDic=read_pickle(aux.path_auxDic)
    for diag in diags:
        base,apps=aux.diag2baps[diag]; dgtp_base=aux.diag2dgtp[base]; dgtp=aux.diag2dgtp[diag]
        isBW=base.endswith('_bw') # PT symmetry will be applied on all apps when doing backward
        bwSgn=-1 if isBW else 1
        if isBW:
            assert('j' not in dgtp_base)
        
        outfile=outpath+diag+'.h5'; outfile_bk=outfile+'_backup'
        files=[file for file in os.listdir(inpath) if file.startswith(base+'.h5')]
        
        # init
        if len(files)==0:
            continue
        if os.path.isfile(outfile_bk):
            os.replace(outfile_bk,outfile)
        outfile_flag_init=outfile+'_flag_init'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag_init):
            with open(outfile_flag_init,'w') as f:
                pass
            with h5py.File(outfile,'w') as f:
                f.create_group('/srcs')
            os.remove(outfile_flag_init)
        # check if skip
        with h5py.File(outfile) as f:
            records=list(f['srcs'].keys())
            Nsrc_pre=np.sum([len(f['srcs'][record]) for record in records])
        files = set(files) - set(records)
        if len(files)==0:
            continue

        # main
        Nsrc=0; dat=0; srcsDic={}
        for file in files:
            with h5py.File(inpath+file) as fbase:
                # print(file,dgtp_base,dgtp)
                srcs=list(fbase['data'].keys()); Nsrc+=len(srcs); srcsDic[file]=srcs
                srcs_int=[src2ints(src) for src in srcs]
                flas= list(fbase['data'][srcs[0]].keys())
                def getDat(src,fla): # expand time
                    tf=aux.Tpack if '_deltat_' not in fla else fla2tf(fla)
                    t=fbase['data'][src][fla][:]
                    return t[[0 if i>tf else i for i in range(aux.Tpack)]]
                t_base=np.array([[getDat(src,fla) for fla in flas] for src in srcs])
                if 'j' not in dgtp_base:
                    t_base=t_base[...,None]
                # t_base.shape=(Nsrc,Nfla,Ntime,Nmom,Nab=4,Ngamma)
                
                dic=aux.moms2dic(fbase['moms'][:])
                moms_full=opabsDic['pavg'][aux.set2key(dgtp)]
                momMap=[dic[tuple(mom)] for mom in aux.moms_full2base(moms_full,dgtp_base)]
                t_base=t_base[:,:,:,momMap]
                
                #===================
                if 'pi0f' in apps:
                    flas_new=[]
                    for fla in flas:
                        if '_deltat_' in fla:
                            a,j,b,dt,tf=fla.split('_')
                            fla_new='_'.join([a+',sgm',j,b,dt,tf])
                            flas_new.append(fla_new)
                        else:
                            a,b=fla.split('_')
                            fla_new='_'.join([a+',sgm',b])
                            flas_new.append(fla_new)
                    flas=flas_new
                    with h5py.File(aux.cfg2post(cfg,'pi0f')) as f:
                        def getApp(st,fla):
                            t=f['data/sgm'][:]
                            timeMap=[(st+bwSgn*fla2tf(fla))%aux.lat_T for tc in range(aux.Tpack)] if 'j' in dgtp_base else\
                                [(st+bwSgn*tf)%aux.lat_T for tf in range(aux.Tpack)]
                            return t[timeMap]
                        dic=aux.moms2dic(f['moms'][:])
                        moms=aux.moms_full2base(moms_full,{'pia'})
                        momMap=[dic[tuple(mom)] for mom in bwSgn*moms]
                        t=np.array([[getApp(src[-1],fla) for fla in flas] for src in srcs_int])
                        t_app=t[:,:,:,momMap]
                        t_phase=np.array([[get_phase(src,mom[6:9]) for mom in bwSgn*moms] for src in srcs_int])
                        t_app=t_app*t_phase[:,None,None,:]
                        t_base=t_base*t_app[:,:,:,:,None,None]
                #===================
                if 'pi0i' in apps:
                    flas_new=[]
                    for fla in flas:
                        if '_deltat_' in fla:
                            a,j,b,dt,tf=fla.split('_')
                            fla_new='_'.join([a,j,b+',sgm',dt,tf])
                            flas_new.append(fla_new)
                        else:
                            a,b=fla.split('_')
                            fla_new='_'.join([a,b+',sgm'])
                            flas_new.append(fla_new)
                    flas=flas_new
                    with h5py.File(aux.cfg2post(cfg,'pi0f')) as f:
                        def getApp(st):
                            t=f['data/sgm'][:]
                            timeMap=[st for t in range(aux.Tpack)]
                            return t[timeMap]
                        dic=aux.moms2dic(moms_exchange_pi(f['moms'][:]))
                        moms=aux.moms_full2base(moms_full,{'pib'})
                        momMap=[dic[tuple(mom)] for mom in bwSgn*moms]
                        t=np.array([getApp(src[-1]) for src in srcs_int])
                        t_app=t[:,:,momMap]
                        t_phase=np.array([[get_phase(src,-mom[0:3]) for mom in bwSgn*moms] for src in srcs_int])
                        t_app=t_app*t_phase[:,None,:]
                        t_base=t_base*t_app[:,None,:,:,None,None]
                #===================
                if 'P' in apps:
                    pis=['sgm']
                    flas_new=[]; flaMap=[]; fla2appky={}
                    for ind_fla,fla in enumerate(flas):
                        for pi in pis:
                            if '_deltat_' in fla:
                                a,j,b,dt,tf=fla.split('_')
                                fla_new='_'.join([a+','+pi,j,b+','+pi,dt,tf])
                                flas_new.append(fla_new); 
                                flaMap.append(ind_fla); fla2appky[fla_new]=pi+'_'+pi
                            else:
                                a,b=fla.split('_')
                                fla_new='_'.join([a+','+pi,b+','+pi])
                                flas_new.append(fla_new); 
                                flaMap.append(ind_fla); fla2appky[fla_new]=pi+'_'+pi
                    flas=flas_new
                    with h5py.File(aux.cfg2post(cfg,'P')) as f:
                        def getApp(st,fla):
                            t=f['data']['st{:03}'.format(st)][fla2appky[fla]][:]
                            timeMap=[(bwSgn*fla2tf(fla))%aux.lat_T for tc in range(aux.Tpack)] if 'j' in dgtp_base else\
                                [(bwSgn*tf)%aux.lat_T for tf in range(aux.Tpack)]
                            return t[timeMap]
                        dic=aux.moms2dic(f['moms'][:])
                        moms=aux.moms_full2base(moms_full,{'pia','pib'})
                        momMap=[dic[tuple(mom)] for mom in bwSgn*moms]
                        t=np.array([[getApp(src[-1],fla) for fla in flas] for src in srcs_int])
                        t_app=t[:,:,:,momMap]
                        t_phase=np.array([[get_phase(src,mom[6:9]-mom[0:3]) for mom in bwSgn*moms] for src in srcs_int])
                        t_app=t_app*t_phase[:,None,None,:]
                        t_base=t_base[:,flaMap]*t_app[:,:,:,:,None,None]
                #===================
                if 'j' in apps:
                    js=['j+','j-','js','jc']
                    flas_new=[]; flaMap=[]; fla2appky={}
                    for ind_fla,fla in enumerate(flas):
                        for j in js:
                            tfs=aux.tfList_disc if j in ['js','jc'] else aux.tfList
                            for tf in tfs:
                                a,b=fla.split('_')
                                fla_new='_'.join([a,j,b,'deltat',str(tf)])
                                flas_new.append(fla_new); 
                                flaMap.append(ind_fla); fla2appky[fla_new]=j
                    flas=flas_new
                    with h5py.File(aux.cfg2post(cfg,'j')) as f:
                        def getApp(st,fla):
                            t=f['data'][fla2appky[fla]][:]
                            timeMap = [(st+bwSgn*tc)%aux.lat_T for tc in range(aux.Tpack)]
                            return t[timeMap]
                        dic=aux.moms2dic(f['moms'][:])
                        moms=aux.moms_full2base(moms_full,{'j'})
                        momMap=[dic[tuple(mom)] for mom in bwSgn*moms]
                        t=np.array([[getApp(src[-1],fla) for fla in flas] for src in srcs_int])
                        t_app=t[:,:,:,momMap]
                        if isBW:
                            t_app=t_app*aux.sgn_PT[None,None,None,None,:]
                        t_phase=np.array([[get_phase(src,mom[9:12]) for mom in bwSgn*moms] for src in srcs_int])
                        t_app=t_app*t_phase[:,None,None,:,None]
                        timeMap_base = [fla2tf(fla) for fla in flas]
                        t_base=t_base[:,flaMap,timeMap_base]
                        t_base=t_base[:,:,None,:,:,:]*t_app[:,:,:,:,None,:]   
                #===================
                elif 'jPi' in apps:
                    js=['j+','j-']
                    flas_new=[]; flaMap=[]; fla2appky={}
                    for ind_fla,fla in enumerate(flas):
                        for tf in aux.tfList:
                            for j in js:
                                a,b=fla.split('_')
                                fla_new='_'.join([a,j,b+',sgm','deltat',str(tf)])
                                flas_new.append(fla_new); 
                                flaMap.append(ind_fla); fla2appky[fla_new]=j+'_sgm'
                    flas=flas_new
                    with h5py.File(aux.cfg2post(cfg,'jPi')) as f:
                        def getApp(st,fla):
                            t=f['data']['st{:03}'.format(st)][fla2appky[fla]][:]
                            timeMap = [(bwSgn*tc)%aux.lat_T for tc in range(aux.Tpack)]
                            return t[timeMap]
                        dic=aux.moms2dic(f['moms'][:])
                        moms=aux.moms_full2base(moms_full,{'j','pib'})
                        momMap=[dic[tuple(mom)] for mom in bwSgn*moms]
                        t=np.array([[getApp(src[-1],fla) for fla in flas] for src in srcs_int])
                        t_app=t[:,:,:,momMap]
                        if isBW:
                            t_app=t_app*aux.sgn_PT[None,None,None,None,:]
                        t_phase=np.array([[get_phase(src,mom[9:12]-mom[0:3]) for mom in bwSgn*moms] for src in srcs_int])
                        t_app=t_app*t_phase[:,None,None,:,None]
                        timeMap_base = [fla2tf(fla) for fla in flas]
                        t_base=t_base[:,flaMap,timeMap_base]
                        t_base=t_base[:,:,None,:,:,:]*t_app[:,:,:,:,None,:]  
                #===================
                elif 'jPf' in apps:
                    js=['j+','j-']
                    flas_new=[]; flaMap=[]; fla2appky={}
                    for ind_fla,fla in enumerate(flas):
                        for tf in aux.tfList:
                            for j in js:
                                a,b=fla.split('_')
                                fla_new='_'.join([a+',sgm',j,b,'deltat',str(tf)])
                                flas_new.append(fla_new); 
                                flaMap.append(ind_fla); fla2appky[fla_new]=j+'_sgm'
                    flas=flas_new
                    with h5py.File(aux.cfg2post(cfg,'jPi')) as f:
                        def getApp(st,fla):
                            t=f['data']['st{:03}'.format((st+bwSgn*fla2tf(fla))%aux.lat_T)][fla2appky[fla]][:]
                            timeMap = [(bwSgn*tc-bwSgn*fla2tf(fla))%aux.lat_T for tc in range(aux.Tpack)]
                            return t[timeMap]
                        dic=aux.moms2dic(moms_exchange_pi(f['moms'][:]))
                        moms=aux.moms_full2base(moms_full,{'pia','j'})
                        momMap=[dic[tuple(mom)] for mom in bwSgn*moms]
                        t=np.array([[getApp(src[-1],fla) for fla in flas] for src in srcs_int])
                        t_app=t[:,:,:,momMap]
                        t_app=t_app
                        if isBW:
                            t_app=t_app*aux.sgn_PT[None,None,None,None,:]
                        t_phase=np.array([[get_phase(src,mom[9:12]+mom[6:9]) for mom in bwSgn*moms] for src in srcs_int])
                        t_app=t_app*t_phase[:,None,None,:,None]
                        timeMap_base = [fla2tf(fla) for fla in flas]
                        t_base=t_base[:,flaMap,timeMap_base]
                        t_base=t_base[:,:,None,:,:,:]*t_app[:,:,:,:,None,:] 
                #===================
                elif 'PJP' in apps:
                    1/0
                
                t_base=np.sum(t_base,axis=0)
                shape=t_base.shape
                shape_new=shape[:2]+(-1,)+shape[4:]
                t=np.reshape(t_base,shape_new)
                tt=auxDic['pavg2avg'][aux.set2key(dgtp)]
                t=sparseMatMul(tt,t,2)
                if 'j' not in dgtp:
                    t=t[...,0]
                dat+=t
            
        # write data
        os.replace(outfile,outfile+'_backup')
        Nsrc+=Nsrc_pre
        with h5py.File(outfile+'_backup') as fr, h5py.File(outfile,'w') as fw:
            fw.create_dataset('opabs',data=opabsDic['avg'][aux.set2key(dgtp)])
            if 'j' in dgtp:
                fw.create_dataset('inserts',data=aux.gjList)
            fw.copy(fr['/srcs'],fw,name='/srcs')
            for file in files:
                fw.create_dataset('/srcs/'+file,data=srcsDic[file])
            for i,fla in enumerate(flas):
                t=dat[i]
                if 'j' in dgtp:
                    t=t[:(fla2tf(fla)+1)]
                if Nsrc_pre!=0:
                    t+=(fr['data'][fla][:]*Nsrc_pre)
                t=t/Nsrc
                fw.create_dataset('/data/'+fla,data=t.astype('complex64'))
        os.remove(outfile+'_backup')
        # break

    print('flag_cfg_done: '+cfg)
                
run()