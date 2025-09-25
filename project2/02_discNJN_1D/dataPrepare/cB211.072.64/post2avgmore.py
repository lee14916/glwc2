'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u post2avgmore.py -c @ > log/post2avgmore.out & 
'''
import h5py,os,re,click
import numpy as np

ens='cB211.072.64'

lat_L={'cB211.072.64':64,'cC211.060.80':80,'cD211.054.96':96,'cE211.044.112':112}[ens]

# max_mom2={'cB211.072.64':23,'cC211.060.80':26,'cD211.054.96':26,'cE211.044.112':4}[ens]
# max_mom2={'cB211.072.64':1,'cC211.060.80':1,'cD211.054.96':1,'cE211.044.112':1}[ens]
max_mom2={'cB211.072.64':14,'cC211.060.80':16,'cD211.054.96':16,'cE211.044.112':4}[ens]
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms_pc=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]

max_mom2=1
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms_pf=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]

moms_target=[pf+pc for pf in moms_pf for pc in moms_pc]
moms_target.sort()
# moms_target=np.array(moms_target)

# tfs={'cB211.072.64':range(2,26+1),'cC211.060.80':range(2,28+1),'cD211.054.96':range(2,32+1),'cE211.044.112':range(2,32+1)}[ens]
tfs={'cB211.072.64':range(2,22+1),'cC211.060.80':range(2,26+1),'cD211.054.96':range(2,30+1),'cE211.044.112':range(2,32+1)}[ens]

stouts=range(0,40+1)
stouts=[4,7,10,13,16,19,22]
# stouts=range(0,4)

def src2ints(src):
    (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
    (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
    return (sx,sy,sz,st)

def get_phase(src_int,mom):
    (sx,sy,sz,st)=src_int
    return np.exp(1j*(2*np.pi/lat_L)*(np.array([sx,sy,sz])@mom))

basepath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run3/{ens}/'

def run_post2avgsrc(cfg):
    inpath=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post/{cfg}/'
    outpath=f'{basepath}data_avgsrc/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    files=[file for file in os.listdir(inpath) if file.startswith('N.h5')]
    
    inpath_fullmom=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_N_fullmom/{cfg}/'
    files_fullmom=[file for file in os.listdir(inpath) if file.startswith('N.h5')]
    outfile=f'{outpath}N.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        
        flas=['N_N']
        data={fla:0 for fla in flas}; data_bw={fla:0 for fla in flas}
        Nsrc=0
        flag_setup=True; moms=[]
        for file in files_fullmom:
            with h5py.File(f'{inpath_fullmom}{file}') as fN:
                if flag_setup:
                    moms=fN[f'moms'][:]
                    flag_setup=False
                NsrcCurrent=len(fN['srcs'])
                Nsrc+=NsrcCurrent
                
                for fla in flas:
                    t=(fN[f'data/N1_N1'][:] + fN[f'data/N2_N2'][:])/2
                    t=np.mean(t,axis=2) * NsrcCurrent
                    data[fla] += t
                    t=(fN[f'data_bw/N1_N1'][:] + fN[f'data_bw/N2_N2'][:])/2
                    t=np.mean(t,axis=2) * NsrcCurrent
                    data_bw[fla] += t
                    
        with h5py.File(outfile,'w') as fw:
            tmax=len(data[flas[0]])-1
            fw.create_dataset('notes',data=['time,mom',f'[time@fwd]=0:{tmax}; [time@bwd]=-{tmax}:-1'])
            fw.create_dataset('moms',data=moms)
            dic={}
            for i,mom in enumerate(moms):
                dic[tuple(mom)]=i
            inds_negmom=[dic[tuple(-mom)] for mom in moms]
            for key,val in data.items():
                fla=key
                # fw.create_dataset(f'data/{fla}',data=data[fla]/Nsrc)    
                # fw.create_dataset(f'data_bw/{fla}',data=data_bw[fla]/Nsrc)    
                
                t=data[fla]/Nsrc
                t_bw=data_bw[fla]/Nsrc
                t_bw=np.flip(t_bw,axis=0)
                t_bw=t_bw[:,inds_negmom]
                t[1:] = (t[1:] + (-1)*t_bw[:])/2
                fw.create_dataset(f'data/{fla}',data=t)    
                
        os.remove(outfile_flag)
    
    with h5py.File(f'{inpath}j.h5') as fj:
        flas=['N_N']
        # js=['j+','j-','js','jc']
        # js=[j for j in fj['data'].keys() if not j.startswith('j-') and ';' in j]
        # js=['j+;g{m,Dn};tl','j-;g{m,Dn};tl','js;g{m,Dn};tl','jc;g{m,Dn};tl']
        js=['j+;g{m,Dn};tl','js;g{m,Dn};tl','jc;g{m,Dn};tl']
        for j in js:
            outfile=f'{outpath}discNJN_{j}.h5'
            outfile_flag=outfile+'_flag'
            if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
                with open(outfile_flag,'w') as f:
                    pass
                
                data={(fla,tf,j):0 for fla in flas for tf in tfs}
                data_bw={(fla,tf,j):0 for fla in flas for tf in tfs}
                Nsrc=0
                for file in files:
                    flag_setup=True
                    with h5py.File(f'{inpath}{file}') as fN:
                        if flag_setup:
                            moms=[list(mom) for mom in fN[f'moms'][:]]
                            momMap_N=[moms.index(mom[:3]) for mom in moms_target]

                            # The loop has exp(-iqx) as the phase, the momentum conservation is p'(sink) + q(transfer) = p(source).
                            # This is opposite to what is used in many ETMC papers.
                            moms_j=[list(mom) for mom in fj[f'moms'][:]]
                            momMap_j=[moms_j.index(mom[-3:]) for mom in moms_target]

                            flag_setup=False
                                
                        for src in fN['data'].keys():
                            src_int=src2ints(src); st=src_int[-1]
                            tPhase=np.array([get_phase(src_int,mom) for mom in moms_j])
                            Nsrc+=1
                            # print(Nsrc,end='                     \r')
                            
                            for fla in flas:
                                datj=fj[f'data/{j}'][:]
                                datj=datj*tPhase[None,:,None]
                                for tf in tfs:
                                    # (time,mom,dirac/proj,insert)
                                    flabase='N_N' if j[:2]!='j-' else 'N_N_-'
                                    tN=fN[f'data/{src}/{flabase}'][tf,:]
                                    tN=tN[momMap_N]
                                    tN=np.transpose(tN[...,None,None],[2,0,1,3])
                                    # print(tN.shape)
                                    tj=np.roll(datj,-st,axis=0)[:tf+1]
                                    tj=np.transpose(tj[:,momMap_j][...,None],[0,1,3,2])
                                    # print(tj.shape)
                                    data[(fla,tf,j)] += tN*tj
                                
                                    # (time,mom,dirac/proj,insert)
                                    tN=fN[f'data_bw/{src}/{flabase}'][-tf,:]
                                    tN=tN[momMap_N]
                                    tN=np.transpose(tN[...,None,None],[2,0,1,3])
                                    # print(tN.shape)
                                    tj=np.roll(datj,-st-1,axis=0)[::-1][:tf+1]
                                    tj=np.transpose(tj[:,momMap_j][...,None],[0,1,3,2])
                                    # print(tj.shape)
                                    data_bw[(fla,tf,j)] += tN*tj
                                        
                            # if Nsrc==10:
                            #     break
                            # break
                
                with h5py.File(outfile,'w') as fw:
                    fw.create_dataset('notes',data=['time,mom,proj,insert','mom=[sink,ins]; sink+ins=src','proj=[P0,Px,Py,Pz]'])
                    fw.create_dataset('moms',data=moms_target)
                    dic={}
                    for i,mom in enumerate(moms_target):
                        dic[tuple(mom)]=i
                    inds_negmom=[dic[tuple(-np.array(mom))] for mom in moms_target]
                    if ';' in j:
                        t=j.split(';')[1:]; t=';'.join(t)
                        ky=f'inserts;{t}'
                    else:
                        ky='inserts'   
                    fw.create_dataset(ky,data=fj[ky][:])
                    for key in data:
                        fla,tf,j=key
                        # fw.create_dataset(f'data/{fla}_{j}_{tf}',data=data[key]/Nsrc)
                        # fw.create_dataset(f'data_bw/{fla}_{j}_{tf}',data=data_bw[key]/Nsrc)
                        
                        assert(j.endswith('g{m,Dn};tl'))
                        t=data[key]/Nsrc
                        t_bw=data_bw[key]/Nsrc
                        t_bw=t_bw[:,inds_negmom]
                        signs=(-1)*np.array([1,-1,-1,-1])
                        t_bw=t_bw*signs[None,None,:,None]
                        t=(t+t_bw)/2
                        fw.create_dataset(f'data/{fla}_{j}_{tf}',data=t)

                os.remove(outfile_flag)
                
        flas=[f'{stout}' for stout in stouts]
        js=['jg;stout']
        for j in js:
            outfile=f'{outpath}discNJN_{j}.h5'
            outfile_flag=outfile+'_flag'
            if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
                with open(outfile_flag,'w') as f:
                    pass
                
                data={(fla,tf,j):0 for fla in flas for tf in tfs}
                data_bw={(fla,tf,j):0 for fla in flas for tf in tfs}
                Nsrc=0
                for file in files:
                    flag_setup=True
                    with h5py.File(f'{inpath}{file}') as fN:
                        if flag_setup:
                            moms=[list(mom) for mom in fN[f'moms'][:]]
                            momMap_N=[moms.index(mom[:3]) for mom in moms_target]

                            # The loop has exp(-iqx) as the phase, the momentum conservation is p'(sink) + q(transfer) = p(source).
                            # This is opposite to what is used in many ETMC papers.
                            moms_j=[list(mom) for mom in fj[f'moms'][:]]
                            momMap_j=[moms_j.index(mom[-3:]) for mom in moms_target]

                            flag_setup=False
                                
                        for src in fN['data'].keys():
                            src_int=src2ints(src); st=src_int[-1]
                            tPhase=np.array([get_phase(src_int,mom) for mom in moms_j])
                            Nsrc+=1
                            # print(Nsrc,end='                     \r')
                            
                            for fla in flas:
                                datj=fj[f'data/{j}{fla}'][:]
                                datj=datj*tPhase[None,:,None]
                                for tf in tfs:
                                    # (time,mom,dirac/proj,insert)
                                    tN=fN[f'data/{src}/N_N'][tf,:]
                                    tN=tN[momMap_N]
                                    tN=np.transpose(tN[...,None,None],[2,0,1,3])
                                    # print(tN.shape)
                                    tj=np.roll(datj,-st,axis=0)[:tf+1]
                                    tj=np.transpose(tj[:,momMap_j][...,None],[0,1,3,2])
                                    # print(tj.shape)
                                    data[(fla,tf,j)] += tN*tj
                                
                                    # (time,mom,dirac/proj,insert)
                                    tN=fN[f'data_bw/{src}/N_N'][-tf,:]
                                    tN=tN[momMap_N]
                                    tN=np.transpose(tN[...,None,None],[2,0,1,3])
                                    # print(tN.shape)
                                    tj=np.roll(datj,-st-1,axis=0)[::-1][:tf+1]
                                    tj=np.transpose(tj[:,momMap_j][...,None],[0,1,3,2])
                                    # print(tj.shape)
                                    data_bw[(fla,tf,j)] += tN*tj
                                        
                            # if Nsrc==10:
                            #     break
                            # break
                
                with h5py.File(outfile,'w') as fw:
                    fw.create_dataset('notes',data=['time,mom,proj,insert','mom=[sink,ins]; sink+ins=src','proj=[P0,Px,Py,Pz]'])
                    fw.create_dataset('moms',data=moms_target)
                    dic={}
                    for i,mom in enumerate(moms_target):
                        dic[tuple(mom)]=i
                    inds_negmom=[dic[tuple(-np.array(mom))] for mom in moms_target]
                    ky='inserts;jg'
                    fw.create_dataset(ky,data=fj[ky][:])
                    for key in data:
                        fla,tf,j=key
                        # fw.create_dataset(f'data/N_N_{j}{fla}_{tf}',data=data[key]/Nsrc)
                        # fw.create_dataset(f'data_bw/N_N_{j}{fla}_{tf}',data=data_bw[key]/Nsrc)
                        
                        t=data[key]/Nsrc
                        t_bw=data_bw[key]/Nsrc
                        t_bw=t_bw[:,inds_negmom]
                        signs=(-1)*np.array([1,-1,-1,-1])
                        t_bw=t_bw*signs[None,None,:,None]
                        t=(t+t_bw)/2
                        fw.create_dataset(f'data/N_N_{j}{fla}_{tf}',data=t)
                            
                os.remove(outfile_flag)
                
from itertools import permutations
from sympy.combinatorics import Permutation
                
def standarizeMom(mom):
    t=np.abs(mom)
    t.sort()
    return t

def run_avgsrc2avgmore(cfg):
    inpath=f'{basepath}data_avgsrc/{cfg}/'
    outpath=f'{basepath}data_avgmore/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    
    outfile=f'{outpath}N.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        
        with h5py.File(f'{inpath}N.h5') as f:
            moms=f['moms'][:]
            dic={}
            for i,mom in enumerate(moms):
                dic[tuple(mom)]=i
            moms_target=list(set([tuple(standarizeMom(mom)) for mom in moms]))
            moms_target.sort()
            moms_target=np.array(moms_target)
            
            elements=[(sx,sy,sz,xyz) for sx in [-1,1] for sy in [-1,1] for sz in [-1,1] for xyz in permutations([0, 1, 2], 3)]
            
            e2inds_mom={}
            for e in elements:
                sx,sy,sz,xyz=e; ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
                e2inds_mom[e]=[dic[(sx*mom[iix],sy*mom[iiy],sz*mom[iiz])] for mom in moms_target]
            
            dat=f['data/N_N'][:]
            t=np.mean([dat[:,e2inds_mom[e]] for e in elements],axis=0)
            
            with h5py.File(outfile,'w') as fw:
                fw.create_dataset('notes',data=f['notes'][:])
                fw.create_dataset('moms',data=moms_target)
                fw.create_dataset('data/N_N',data=t)
            os.remove(outfile_flag)

    flag=True
    for j in ['j+','js','jc','jg'][:]:
        filename=f'discNJN_{j};g{{m,Dn}};tl.h5'
        inserts_name='inserts;g{m,Dn};tl'
        
        if j=='jg':
            filename=f'discNJN_jg;stout.h5'
            inserts_name='inserts;jg'
            
        outfile=f'{outpath}{filename}'
        outfile_flag=outfile+'_flag'
        if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
            with open(outfile_flag,'w') as f:
                pass
            
            with h5py.File(f'{inpath}{filename}') as f:
                if flag:
                    moms=f['moms'][:]
                    dic={}
                    for i,mom in enumerate(moms):
                        dic[tuple(mom)]=i
                    moms_target=list(set([tuple(list(mom[:3])+list(standarizeMom(mom[3:6]))) for mom in moms]))
                    moms_target.sort()
                    moms_target=np.array(moms_target)
                    
                    inserts=[insert.decode() for insert in f[inserts_name][:]]
                    
                    elements=[(sx,sy,sz,xyz) for sx in [1,-1] for sy in [1,-1] for sz in [1,-1] for xyz in permutations([0, 1, 2], 3)]
                    
                    e2inds_mom={}; e2inds_proj={}; e2signs_proj={}; e2inds_insert={}; e2signs_insert={}
                    for e in elements:
                        sx,sy,sz,xyz=e; signs=[sx,sy,sz,1]
                        ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
                        xyzt=['x','y','z','t']
                        xyzt2={'x':xyzt[ix],'y':xyzt[iy],'z':xyzt[iz],'t':'t'}
                        det=sx*sy*sz*(1 if Permutation(xyz).is_even else -1)
                        
                        e2inds_mom[e]=[dic[(sx*mom[iix],sy*mom[iiy],sz*mom[iiz],sx*mom[iix+3],sy*mom[iiy+3],sz*mom[iiz+3])] for mom in moms_target]
                        
                        e2signs_proj[e]=np.array([1,sx*det,sy*det,sz*det])[None,None,:,None]
                        e2inds_proj[e]=[0,ix+1,iy+1,iz+1]
                        
                        e2signs_insert[e]=np.array([signs[xyzt.index(insert[0])]*signs[xyzt.index(insert[1])] for insert in inserts])[None,None,None,:]
                        e2inds_insert[e]=[xyzt2[insert[0]]+xyzt2[insert[1]] for insert in inserts]
                        e2inds_insert[e]=[inserts.index(ele) if ele in inserts else inserts.index(ele[1]+ele[0]) for ele in e2inds_insert[e]]
                    flag=False
            
                
                with h5py.File(outfile,'w') as fw:
                    fw.create_dataset('notes',data=f['notes'][:])
                    fw.create_dataset(inserts_name,data=f[inserts_name][:])
                    fw.create_dataset('moms',data=moms_target)
                    
                    for key in f['data'].keys():
                        dat=f['data'][key][:]
                        
                        def get(e):
                            t=dat[:,e2inds_mom[e]]
                            
                            t=t*e2signs_proj[e]
                            t=t[:,:,e2inds_proj[e]]
                            
                            t=t*e2signs_insert[e]
                            t=t[:,:,:,e2inds_insert[e]]
                            
                            # print(e,np.real(t[0,imom,iproj,iins]),moms[e2inds_mom[e][imom]],e2signs_proj[e][0,0,iproj,0],e2inds_proj[e][iproj],e2signs_insert[e][0,0,0,iins],e2inds_insert[e][iins])
                            return t
                        

                        # print(moms_target[imom])
                        # t=np.array([get(e) for e in elements])
                        # print(t.shape)
                        # t=np.real(t[:,0,imom,iproj,iins])
                        # print(np.mean(t))
                        # print(t)
                        
                        t=np.mean([get(e) for e in elements],axis=0)
                        fw.create_dataset(f'data/{key}',data=t)
                os.remove(outfile_flag)

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    run_post2avgsrc(cfg)
    run_avgsrc2avgmore(cfg)
    
    path=f'{basepath}data_avgsrc/{cfg}/'
    for file in os.listdir(path):
        with h5py.File(f'{path}{file}','w') as f:
            pass
    
    print('flag_cfg_done: '+cfg)

run()