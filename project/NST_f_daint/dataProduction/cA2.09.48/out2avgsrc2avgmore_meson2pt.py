'''
cat data_aux/cfgs_run | xargs -n 1 -I @ -P 10 python3 -u out2avgsrc2avgmore_meson2pt.py -c @ > log/out2avgsrc2more_meson2pt.out & 
'''
import os, click, h5py, re, pickle
import numpy as np


def cfg2out_jPP(cfg):
    path='/project/s1174/lyan/code/scratch/run/NST_f/cA2.09.48/jPP/out/'+cfg+'/'
    return path

def cfg2out_loop(cfg):
    path='/project/s1174/lyan/code/scratch/run/NST_f/cA2.09.48/loop/out/'+cfg+'/'
    return path

flavorRecombination={ # (coe,fla,gmf,gmi) # A means axial # S/L means smear/local
    'pi+':[(1j,'dBuS','g5')], 'pi-':[(1j,'uBdS','g5')],
    'pi+L':[(1j,'dBuL','g5')], 'pi-L':[(1j,'uBdL','g5')],
    'pi+A':[(1j,'dBuS','g5gt')], 'pi-A':[(1j,'uBdS','g5gt')],
    'pi+AL':[(1j,'dBuL','g5gt')], 'pi-AL':[(1j,'uBdL','g5gt')],
    
    'pi0':[(1j/np.sqrt(2),'uBuS','g5'),(-1j/np.sqrt(2),'dBdS','g5')],
    'pi0L':[(1j/np.sqrt(2),'uBuL','g5'),(-1j/np.sqrt(2),'dBdL','g5')],
    'pi0A':[(1j/np.sqrt(2),'uBuS','g5gt'),(-1j/np.sqrt(2),'dBdS','g5gt')],
    'pi0AL':[(1j/np.sqrt(2),'uBuL','g5gt'),(-1j/np.sqrt(2),'dBdL','g5gt')],
                               
    'sgm':[(1/np.sqrt(2),'uBuS','id'),(1/np.sqrt(2),'dBdS','id')],
    'sgmL':[(1/np.sqrt(2),'uBuL','id'),(1/np.sqrt(2),'dBdL','id')],
}

def flavorChange(fla): # f1Bf2 -> {f2}B{f1}
    assert(len(fla)==4 and fla[1]=='B' and fla[3] in ['S','L'])
    dic={'u':'d','d':'u'}
    t=dic[fla[2]]+fla[1]+dic[fla[0]]+fla[3]
    return t

Psgn={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':-1,'sgmxy':1,'sgmyz':1,'sgmzx':1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}
gtCj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':1,'sgmty':1,'sgmtz':1}
g5Cj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':-1,'g5':1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1}

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    assert(len(cfg[1:])==4 and cfg[1:].isdigit())
    outpath='data_avgsrc/'+cfg+'/'
    os.makedirs(outpath,exist_ok=True)

    outfile=outpath+'P.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw:
            infile=cfg2out_jPP(cfg)+'jPP.h5'
            with h5py.File(infile) as fr:
                moms=fr['moms'][:]
                moms_new0=[]; moms_map=[]
                for i,mom in enumerate(moms):
                    if np.all(mom[:3] == mom[3:]):
                        moms_new0.append(mom[:3])
                        moms_map.append(i)
                        
                fw.create_dataset('moms',data=moms_new0)

                srcs=list(fr['data'].keys()); srcs.sort()
                flas=list(fr['data'][srcs[0]].keys()); flas.sort()
                
                dic_Gas={}
                for i,Ga in enumerate(fr['Gfs'][:]):
                    dic_Gas[Ga.decode()]=i
                dic_Gbs={}
                for i,Gb in enumerate(fr['Gis'][:]):
                    dic_Gbs[Gb.decode()]=i
                
                dat={}
                for fla in flas:
                    t=np.mean([fr['data'][src][fla] for src in srcs],axis=0)
                    t=t[:,moms_map,:,:]
                    dat[fla]=t
                
                for fla0 in ['pi+','pi-','pi0']:
                    flas=[fla0+f for f in ['','L','A','AL']]
                    if fla0 == 'pi0':
                        flas += ['sgm','sgmL']
                    for flaa in flas:
                        for flab in flas:
                            if flaa[-1]!= 'L' and flab[-1]=='L':
                                continue
                            
                            t=0
                            for ca,fa,ga in flavorRecombination[flaa]:
                                for cb,fb,gb in flavorRecombination[flab]:
                                    if fa[0]!=fb[0]:
                                        continue
                                    fab=fa+'_'+fb
                                    t+=ca*np.conj(cb)*dat[fab][:,:,dic_Gas[ga],dic_Gbs[gb]]
                            flaab=flaa+'_'+flab
                            fw.create_dataset(f'data/{flaab}',data=t.astype('complex64'))
        os.remove(outfile_flag)
                                
    outfile=outpath+'pi0f-pi0i.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw:
            infileS=cfg2out_loop(cfg)+'loopS.h5'
            infileL=cfg2out_loop(cfg)+'loopL.h5'
            with h5py.File(infileS) as fs, h5py.File(infileL) as fl: 
                dat={}
                dic_Gs={'L':{},'S':{}}
                
                moms=fl['moms'][:]
                momsS=fs['moms'][:]
                assert(np.all(moms==momsS))
                dic={}; 
                for i,mom in enumerate(moms):
                    dic[tuple(mom)]=i
                    if np.all(mom==[0,0,0]):
                        ind_mom0=i
                moms_nega_map=[dic[tuple(-mom)] for mom in moms]
                
                fw.create_dataset('moms',data=moms)
                
                sign_Gs=np.array([g5Cj[G.decode()] for G in fl['Gs'][:]])
                dat['uBuL']=np.mean([np.array(fl['data'][seed][id]['up'][:]) for seed in fl['data'].keys() for id in fl['data'][seed].keys()],axis=0)
                dat['dBdL']=np.conj(dat['uBuL'][:,moms_nega_map,:]*sign_Gs[None,None,:])
                for i,G in enumerate(fl['Gs'][:]):
                    dic_Gs['L'][G.decode()]=i
                
                sign_Gs=np.array([g5Cj[G.decode()] for G in fs['Gs'][:]])
                dat['uBuS']=np.mean([np.array(fs['data'][seed][id]['up'][:]) for seed in fs['data'].keys() for id in fs['data'][seed].keys()],axis=0)
                dat['dBdS']=np.conj(dat['uBuS'][:,moms_nega_map,:]*sign_Gs[None,None,:])
                for i,G in enumerate(fs['Gs'][:]):
                    dic_Gs['S'][G.decode()]=i
          
                for fla0 in ['pi0']:
                    flas=[fla0+f for f in ['','L','A','AL']]
                    if fla0 == 'pi0':
                        flas += ['sgm','sgmL']
                    for flaa in flas:
                        for flab in flas:
                            t=0
                            for ca,fa,ga in flavorRecombination[flaa]:
                                for cb,fb,gb in flavorRecombination[flab]:
                                    ta=dat[fa][:,:,dic_Gs[fa[-1]][ga]]; tb=dat[fb][:,:,dic_Gs[fb[-1]][gb]]
                                    
                                    tb=tb[:,moms_nega_map]*gtCj[gb]
                                    tt=np.mean([np.roll(ta,-st,axis=0)*tb[st:st+1] for st in range(len(tb))],axis=0)
                                    t+= ca*np.conj(cb)*tt
                                    # print(type(t[0,0]))
                            flaab=flaa+'_'+flab
                            fw.create_dataset(f'data/{flaab}',data=t)
                            # print(fla0,flaa,flab,type(t[0,0]))
                            
                    for fla in flas:
                        t=0
                        for c,f,g in flavorRecombination[fla]:
                            tt=dat[f][:,ind_mom0,dic_Gs[f[-1]][g]]
                            t+= c*np.mean(tt,axis=0)
                        fw.create_dataset(f'VEV/{fla}',data=t)
                        # print(t.shape,type(t))
                    
        os.remove(outfile_flag)
        
        
    inpath=f'data_avgsrc/{cfg}/'
    outpath=f'data_avgmore/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    
    for diag in ['P','pi0f-pi0i']:
        infile=inpath+diag+'.h5'
        outfile=outpath+diag+'.h5'
        with h5py.File(outfile,'w') as fw, h5py.File(infile) as fr:
            dat={}
            for fla in fr['data'].keys():
                dat[fla]=fr['data'][fla][:]
            
            flas=list(dat.keys())
            flas.sort()
            # flavor exchange * parity * rotate back (twisted flavor exchange) [ average pi+/-; nothing to do with pi0 & sgm ]
            for fla in flas:
                flaa,flab=fla.split('_')
                
                if 'pi+' in flaa:
                    fla2=fla.replace('pi+','pi-')
                    dat[fla]=(dat[fla]+dat[fla2])/2
                    del dat[fla2]
                    
            # rotation
            moms=fr['moms'][:]
            inds0=[]; inds1=[]
            for i,mom in enumerate(moms):
                mom2=mom[0]**2+mom[1]**2+mom[2]**2
                if mom2 == 0:
                    inds0.append(i)
                elif mom2 == 1:
                    inds1.append(i)
            indss=[inds0,inds1]
            fw.create_dataset('moms',data=[[0,0,0],[0,0,1]])
        
            for fla in dat.keys():
                dat[fla]=np.transpose([np.mean(dat[fla][:,inds],axis=1) for inds in indss],[1,0])
            
            for fla in dat.keys():
                fw.create_dataset(f'data/{fla}',data=dat[fla])
                
            if 'VEV' in fr.keys():
                for fla in fr['VEV'].keys():
                    fw.create_dataset(f'VEV/{fla}',data=fr['VEV'][fla][()])

    print('flag_cfg_done: '+cfg)

run()