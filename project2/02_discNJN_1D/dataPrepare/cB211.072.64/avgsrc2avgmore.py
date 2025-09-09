'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u avgsrc2avgmore.py -c @ > log/avgsrc2avgmore.out & 
'''
import h5py,os,re,click
import numpy as np
from itertools import permutations

ens='cB211.072.64'

lat_L={'cB211.072.64':64,'cC211.060.80':80,'cD211.054.96':96,'cE211.044.112':112}[ens]

# max_mom2={'cB211.072.64':23,'cC211.060.80':26,'cD211.054.96':26,'cE211.044.112':4}[ens]
# max_mom2={'cB211.072.64':1,'cC211.060.80':1,'cD211.054.96':1,'cE211.044.112':1}[ens]
max_mom2={'cB211.072.64':14,'cC211.060.80':16,'cD211.054.96':16,'cE211.044.112':4}[ens]
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms_pc=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]

max_mom2=0
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

def standarizeMom(mom):
    t=np.abs(mom)
    t.sort()
    return t

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    basepath=f'/p/project1/ngff/li47/code/scratch/run/02_discNJN_1D_run2/{ens}/'
    inpath=f'{basepath}/data_avgsrc/{cfg}/'
    outpath=f'{basepath}/data_avgmore/{cfg}/'
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
    for j in ['j+','js','jc','jg']:
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
                    
                    elements=[(sx,sy,sz,xyz) for sx in [-1,1] for sy in [-1,1] for sz in [-1,1] for xyz in permutations([0, 1, 2], 3)]
                    
                    e2inds_mom={}; e2inds_proj={}; e2signs_proj={}; e2inds_insert={}; e2signs_insert={}
                    for e in elements:
                        sx,sy,sz,xyz=e; signs=[sx,sy,sz,1]
                        ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
                        
                        xyzt=['x','y','z','t']
                        xyzt2={'x':xyzt[ix],'y':xyzt[iy],'z':xyzt[iz],'t':'t'}
                        
                        e2inds_mom[e]=[dic[(mom[0],mom[1],mom[2],sx*mom[iix+3],sy*mom[iiy+3],sz*mom[iiz+3])] for mom in moms_target]
                        
                        e2signs_proj[e]=np.array([1,sx,sy,sz])[None,None,:,None]
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
                            return t
                        
                        t=np.mean([get(e) for e in elements],axis=0)
                        fw.create_dataset(f'data/{key}',data=t)
                os.remove(outfile_flag)
                
    print('flag_cfg_done: '+cfg)
            
run()