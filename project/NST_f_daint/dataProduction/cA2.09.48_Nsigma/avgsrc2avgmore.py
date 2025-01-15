'''
cat data_aux/cfgs_run | xargs -n 1 -I @ -P 10 python3 -u avgsrc2avgmore.py -c @ > log/avgsrc2avgmore.out & 
'''

import os, click, h5py, re
import numpy as np
import pickle
import aux
from scipy.sparse import csr_matrix

def fla2tf(fla):
    assert('_deltat_' in fla)
    return int(fla.split('_deltat_')[-1])
def getDat(fla,tfile): # expand time
    tf=aux.Tpack if '_deltat_' not in fla else fla2tf(fla)
    t=tfile['data'][fla][:]
    return t[[0 if i>tf else i for i in range(aux.Tpack)]]

def sparseMatMul(sm,nda,axis):
    t=np.moveaxis(nda,axis,0)
    ts=t.shape
    t=np.reshape(t,(ts[0],-1))
    t=sm@t
    t=np.reshape(t,(-1,)+ts[1:])
    t=np.moveaxis(t,0,axis)
    return t

with open('data_aux/avgDirection.pkl','rb') as f:
    avgD=pickle.load(f)
    
def rotate(opabj,rot):
    if rot in ['0,0,0','0,0,1']:
        return opabj
    
    # 2pt
    if len(opabj[0])==2:
        temp=[]
        for coe,opab in opabj:
            opa,opb=opab.split('_')
            g,pt,irrep,occa,l,flaa=opa.split(';'); g,pt,irrep,occb,l,flab=opb.split(';')
            opa2=';'.join([g,rot,irrep,occa,l,flaa]); opb2=';'.join([g,rot,irrep,occb,l,flab])
            opab2=opa2+'_'+opb2
            temp.append((coe,opab2))
        return temp
    
    # 3pt
    temp=[] # rotate insert
    for coe,opab,insert in opabj:
        gm,j,tf=insert.split('_')
        if gm in ['id','gt','g5','g5gt']:
            temp.append((coe,opab,insert))
        elif gm in ['gx','gy','gz']:
            i_gm={'gx':0,'gy':1,'gz':2}[gm]
            for j_gm,val in enumerate(avgD[rot]['gamma_i'][i_gm,:]):
                if np.abs(val)<1e-7:
                    continue
                insert_new='_'.join([['gx','gy','gz'][j_gm],j,tf])
                temp.append((coe*val,opab,insert_new))
        elif gm in ['g5gx','g5gy','g5gz']:
            i_gm={'g5gx':0,'g5gy':1,'g5gz':2}[gm]
            for j_gm,val in enumerate(avgD[rot]['gamma_i'][i_gm,:]):
                if np.abs(val)<1e-7:
                    continue
                insert_new='_'.join([['g5gx','g5gy','g5gz'][j_gm],j,tf])
                temp.append((coe*val,opab,insert_new))
        elif gm in ['sgmyz','sgmzx','sgmxy']:
            i_gm={'sgmyz':0,'sgmzx':1,'sgmxy':2}[gm]
            for j_gm,val in enumerate(avgD[rot]['gamma_i'][i_gm,:]):
                if np.abs(val)<1e-7:
                    continue
                insert_new='_'.join([['sgmyz','sgmzx','sgmxy'][j_gm],j,tf])
                temp.append((coe*val,opab,insert_new))
        elif gm in ['sgmtx','sgmty','sgmtz']:
            i_gm={'sgmtx':0,'sgmty':1,'sgmtz':2}[gm]
            for j_gm,val in enumerate(avgD[rot]['gamma_i'][i_gm,:]):
                if np.abs(val)<1e-7:
                    continue
                insert_new='_'.join([['sgmtx','sgmty','sgmtz'][j_gm],j,tf])
                temp.append((coe*val,opab,insert_new))
        else:
            1/0
            
    temp2=[] # rotate opa
    for coe,opab,insert in temp:
        opa,opb=opab.split('_')
        g,pt,irrep,occ,lam,fla=opa.split(';')
        assert(pt in ['0,0,0','0,0,1'])
        if pt !='0,0,0':
            opa_new=';'.join([g,rot,irrep,occ,lam,fla])
            temp2.append((coe,'_'.join([opa_new,opb]),insert))
        else:
            i_lam={'l1':0,'l2':1}[lam]
            for j_lam,val in enumerate(avgD[rot]['irrep_row'][i_lam,:]):
                if np.abs(val)<1e-7:
                    continue
                opa_new=';'.join([g,pt,irrep,occ,['l1','l2'][j_lam],fla])
                temp2.append((coe*val,'_'.join([opa_new,opb]),insert))
                
    temp3=[] # rotate opb
    for coe,opab,insert in temp2:
        opa,opb=opab.split('_')
        g,pt,irrep,occ,lam,fla=opb.split(';')
        assert(pt in ['0,0,0','0,0,1'])
        if pt !='0,0,0':
            opb_new=';'.join([g,rot,irrep,occ,lam,fla])
            temp3.append((coe,'_'.join([opa,opb_new]),insert))
        else:
            i_lam={'l1':0,'l2':1}[lam]
            for j_lam,val in enumerate(avgD[rot]['irrep_row'][i_lam,:]):
                if np.abs(val)<1e-7:
                    continue
                opb_new=';'.join([g,pt,irrep,occ,['l1','l2'][j_lam],fla])
                temp3.append((coe*np.conj(val),'_'.join([opa,opb_new]),insert))

    return temp3

def PTrans0(opab):
    '''
    Here may not work for general irreps.
    '''
    opa,opb=opab.split('_')
    sgn=1
    
    g,pt,irrep,occ,lam,fla=opa.split(';')
    assert(fla in ['N','N,pi','N,sgm'])
    if pt=='0,0,0':
        sgn*={'G1g':1,'G1u':-1}[irrep]
        opa_new=opa
    else:
        px,py,pz=pt.split(','); px,py,pz=str(-int(px)),str(-int(py)),str(-int(pz)); 
        pt_new=','.join([px,py,pz])
        lam_new={'l1':'l2','l2':'l1'}[lam]
        opa_new=';'.join([g,pt_new,irrep,occ,lam_new,fla])
        sgn*=avgD['P'][pt][lam]
        
    g,pt,irrep,occ,lam,fla=opb.split(';')
    assert(fla in ['N','N,pi','N,sgm'])
    if pt=='0,0,0':
        sgn*={'G1g':1,'G1u':-1}[irrep]
        opb_new=opb
    else:
        px,py,pz=pt.split(','); px,py,pz=str(-int(px)),str(-int(py)),str(-int(pz)); 
        pt_new=','.join([px,py,pz])
        lam_new={'l1':'l2','l2':'l1'}[lam]
        opb_new=';'.join([g,pt_new,irrep,occ,lam_new,fla])
        sgn*=np.conj(avgD['P'][pt][lam])

    opab_new=opa_new+'_'+opb_new
    return sgn,opab_new

def PTrans(opabs):   
    dic={}
    for i,opab in enumerate(opabs):
        dic[opab]=i
    t=[PTrans0(opab) for opab in opabs]
    opabsMap=np.array([dic[opab] for sgn,opab in t])
    sgns=np.array([sgn for sgn,opab in t])
    return opabsMap,sgns
    
flaFlip={'p':(1,'n'),'p,pi0':(-1,'n,pi0'),'n,pi+':(1,'p,pi-'),'p,sgm':(1,'n,sgm')}
mainFlas=flaFlip.keys()

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath='data_avgsrc/'+cfg+'/'
    outpath='data_avgmore/'+cfg+'/'
    
    with open(aux.path_diags_all,'r') as f:
        diags_all=f.read().splitlines()
    
    # load data
    data={}
    for diag in diags_all:
        data[diag]={}
        infile=inpath+diag+'.h5'
        with h5py.File(infile) as f:
            datasets=[]
            def visit_function(name,node):
                if isinstance(node, h5py.Dataset):
                    datasets.append(name)
                    # print(len(datasets),name,end='\r')
            f.visititems(visit_function)
            for dataset in datasets:
                if dataset.startswith('srcs/') or dataset in ['inserts','opabs']:
                    t=f[dataset][()]
                    data[diag][dataset]=[ele.decode() for ele in t]
                else:
                    data[diag][dataset]=f[dataset][()]

    # flavors (flavor exchaning * parity transformation)
    toDelete=[]
    for diag in data.keys():
        for dataset in data[diag].keys():
            if not dataset.startswith('data'):
                continue
            data_str,fla=dataset.split('/')
            if '_deltat_' not in fla:
                flaa,flab=fla.split('_')
            else:
                flaa,j,flab,dt,tf=fla.split('_')
            if not (flaa in mainFlas and flab in mainFlas):
                toDelete.append([diag,dataset])
                continue
            sgna,flaa_new=flaFlip[flaa]
            sgnb,flab_new=flaFlip[flab]
            sgn=sgna*np.conj(sgnb)
            
            if '_deltat_' not in fla:
                fla_new='_'.join([flaa_new,flab_new])
                dataset_new='/'.join([data_str,fla_new])
                if dataset_new not in data[diag].keys():
                    continue
                opabsMap,sgns_opab=PTrans(data[diag]['opabs'])
                data[diag][dataset]=(data[diag][dataset]+sgn*data[diag][dataset_new][:,opabsMap]*sgns_opab[None,:])/2
            else:
                sgn*={'j+':1,'j-':-1}[j]
                fla_new='_'.join([flaa_new,j,flab_new,dt,tf])
                dataset_new='/'.join([data_str,fla_new])
                if dataset_new not in data[diag].keys():
                    continue
                opabsMap,sgns_opab=PTrans(data[diag]['opabs'])
                sgns_insert=aux.sgn_P
                t=sgn*data[diag][dataset_new][:,opabsMap,:]*sgns_opab[None,:,None]
                t=t[:,:,:]*sgns_insert[None,None,:]
                data[diag][dataset]=(data[diag][dataset]+t)/2
    for diag,dataset in toDelete:
        del data[diag][dataset]
     
    # rotations (proper rotations)
    pts_final=['0,0,0','0,0,1']
    data_new={}
    for diag in data.keys():
        # non-data dataset for data_new
        data_new[diag]={}
        for dataset in data[diag].keys():
            if dataset.startswith('srcs/') or dataset=='inserts':
                data_new[diag][dataset]=data[diag][dataset]
                continue
            if dataset=='opabs':
                t=data[diag][dataset]
                data_new[diag][dataset]=[]
                for i,opab in enumerate(t):
                    opa,opb=opab.split('_')
                    _,pta,_,_,_,_=opa.split(';'); _,ptb,_,_,_,_=opb.split(';')
                    if (pta not in pts_final) or (ptb not in pts_final):
                        continue
                    data_new[diag][dataset].append(opab)
                data_new[diag][dataset].sort()
        
        # build avg matrix
        vals=[]; row=[]; col=[]
        if 'j' not in aux.diag2dgtp[diag]:
            dic={}
            for i,opab in enumerate(data[diag]['opabs']):
                dic[opab]=i
            for i,opab_new in enumerate(data_new[diag]['opabs']):
                pt=opab_new.split('_')[0].split(';')[1]
                if pt=='0,0,0':
                    vals.append(1); row.append(i); col.append(dic[opab_new])
                elif pt=='0,0,1':
                    opabj_base=[(1,opab_new)]
                    for rot in ['0,0,1','0,0,-1','0,1,0','0,-1,0','1,0,0','-1,0,0']:
                        for coe,opab in rotate(opabj_base,rot):
                            vals.append(coe/6); row.append(i); col.append(dic[opab])
            shape=(len(data_new[diag]['opabs']), len(data[diag]['opabs']))
            avgM=csr_matrix((vals, (row, col)), shape=shape)
            avgM.eliminate_zeros()
        else:
            dic={}
            for i,opab in enumerate(data[diag]['opabs']):
                for j,gm in enumerate(data[diag]['inserts']):
                    dic[(opab,gm)]=i*len(data[diag]['inserts'])+j
            for i,opab_new in enumerate(data_new[diag]['opabs']):
                for j,gm_new in enumerate(data_new[diag]['inserts']):
                    opa,opb=opab_new.split('_')
                    pta=opa.split(';')[1]; ptb=opb.split(';')[1]
                    assert(pta in ['0,0,0','0,0,1'] and ptb in ['0,0,0','0,0,1'])
                    opabj_base=[(1,opab_new,'_'.join([gm_new,'j+','10']))]
                    for rot in ['0,0,1','0,0,-1','0,1,0','0,-1,0','1,0,0','-1,0,0']:
                        for coe,opab,insert in rotate(opabj_base,rot):
                            vals.append(coe/6); row.append(i*len(data_new[diag]['inserts'])+j); col.append(dic[(opab,insert.split('_')[0])])
            shape=(len(data_new[diag]['opabs']*len(data_new[diag]['inserts'])), len(data[diag]['opabs'])*len(data[diag]['inserts']))
            avgM=csr_matrix((vals, (row, col)), shape=shape)
            avgM.eliminate_zeros()
                
        for dataset in data[diag].keys():
            if not dataset.startswith('data'):
                continue
            if 'j' not in aux.diag2dgtp[diag]:
                data_new[diag][dataset]=sparseMatMul(avgM,data[diag][dataset],1)
            else:
                t=data[diag][dataset]
                t=sparseMatMul(avgM,np.reshape(t,(t.shape[0],-1)),1)
                data_new[diag][dataset]=np.reshape(t,(t.shape[0],-1,len(data_new[diag]['inserts'])))
    data=data_new
    
    # backward (PT transformation)
    for diag in diags_all:
        base,apps=aux.diag2baps[diag]
        if not base.endswith('_bw'):
            continue
        diag_fw='-'.join([base[:-3]]+apps)
        for dataset in data[diag_fw].keys():
            if not dataset.startswith('data'):
                continue
            data[diag_fw][dataset]=(data[diag_fw][dataset]+data[diag][dataset])/2
        del data[diag]
    
    # save
    os.makedirs(outpath,exist_ok=True)
    for diag in data.keys():
        outfile=outpath+diag+'.h5'
        with h5py.File(outfile,'w') as f:
            for dataset in data[diag].keys():
                t=data[diag][dataset]
                f.create_dataset(dataset,data=t.astype('complex64') if dataset.startswith('data') else t)
        
    print('flag_cfg_done: '+cfg)
                
run()