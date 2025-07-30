'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u post2avgsrc_scalingCheck.py -c @ > log/post2avgsrc_scalingCheck.out & 
'''
import h5py,os,re,click
import numpy as np
from random import shuffle

ens='cC211.060.80'

lat_L={'cB211.072.64':64,'cC211.060.80':80,'cD211.054.96':96}[ens]

max_mom2={'cB211.072.64':23,'cC211.060.80':1,'cD211.054.96':26}[ens]
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms_pc=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]
max_mom2=1
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms_pf=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]

# moms_target=[pf+pc for pf in moms_pf for pc in moms_pc]
moms_target=[pf+[0,0,0] for pf in moms_pf]
moms_target.sort()
# moms_target=np.array(moms_target)

tfs={'cB211.072.64':range(2,26+1),'cC211.060.80':range(2,20+1,2),'cD211.054.96':range(2,32+1)}[ens]

def src2ints(src):
    (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
    (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
    return (sx,sy,sz,st)

def get_phase(src_int,mom):
    (sx,sy,sz,st)=src_int
    return np.exp(1j*(2*np.pi/lat_L)*(np.array([sx,sy,sz])@mom))

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'/p/project/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post_scalingCheck/{cfg}/'
    outpath=f'/p/project/ngff/li47/code/scratch/run/02_discNJN_1D/{ens}/data_avgsrc_scalingCheck/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    files=[file for file in os.listdir(inpath) if file.startswith('N.h5')]
    
    with h5py.File(f'{inpath}j.h5') as fj:
        flas=['N_N']
        # js=['j+','j-','js','jc']
        # js=[j for j in fj['data'].keys() if not j.startswith('j-') and ';' in j]
        js=['j+;g{m,Dn};tl','js;g{m,Dn};tl','jc;g{m,Dn};tl'][1:2]
        for j in js:
            outfile=f'{outpath}discNJN_{j}.h5'
            outfile_flag=outfile+'_flag'
            if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
                with open(outfile_flag,'w') as f:
                    pass
                
                with h5py.File(outfile,'w') as fw:
                    file2srcs={}
                    for t_data in ['data1','data2','data3','data4']:
                        if t_data=='data1':
                            dataN=[]
                            dataN_bw=[]
                        data={(fla,tf,j):[] for fla in flas for tf in tfs}
                        data_bw={(fla,tf,j):[] for fla in flas for tf in tfs}
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

                                if t_data=='data1':
                                    srcs=list(fN['data'].keys())
                                    shuffle(srcs)
                                    assert(file not in file2srcs)
                                    file2srcs[file]=srcs.copy()
                                
                                srcs=file2srcs[file]
                                    
                                for src in srcs:
                                    src_int=src2ints(src); st=src_int[-1]
                                    tPhase=np.array([get_phase(src_int,mom) for mom in moms_j])
                                    Nsrc+=1
                                    # print(Nsrc,end='                     \r')
                                    
                                    if t_data=='data1':
                                        t=fN[f'data/{src}/N_N'][:]
                                        dataN.append(t[:,momMap_N])
                                        t=fN[f'data_bw/{src}/N_N'][:]
                                        dataN_bw.append(t[:,momMap_N])

                                    for fla in flas:
                                        datj=fj[f'{t_data}/{j}'][:]
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
                                            data[(fla,tf,j)].append(tN*tj)
                                        
                                            # (time,mom,dirac/proj,insert)
                                            tN=fN[f'data_bw/{src}/N_N'][-tf,:]
                                            tN=tN[momMap_N]
                                            tN=np.transpose(tN[...,None,None],[2,0,1,3])
                                            # print(tN.shape)
                                            tj=np.roll(datj,-st-1,axis=0)[::-1][:tf+1]
                                            tj=np.transpose(tj[:,momMap_j][...,None],[0,1,3,2])
                                            # print(tj.shape)
                                            data_bw[(fla,tf,j)].append(tN*tj)
                                                
                                    # if Nsrc==10:
                                    #     break
                                    # break
                        
                        if t_data=='data1':
                            fw.create_dataset('notes',data=['time,mom,proj,insert','mom=[sink,ins]; sink+ins=src','proj=[P0,Px,Py,Pz]'])
                            fw.create_dataset('moms',data=moms_target)
                            if ';' in j:
                                t=j.split(';')[1:]; t=';'.join(t)
                                ky=f'inserts;{t}'
                            else:
                                ky='inserts'
                            fw.create_dataset(ky,data=fj[ky][:4])
                        
                        if t_data=='data1':
                            fw.create_dataset(f'dataN',data=dataN)
                            fw.create_dataset(f'dataN_bw',data=dataN_bw)
                        for key in data:
                            fla,tf,j=key
                            t=np.array(data[key])
                            fw.create_dataset(f'{t_data}/{fla}_{j}_{tf}',data=t[:,:,:,:1,:4], dtype='complex64')
                            t=np.array(data_bw[key])
                            fw.create_dataset(f'{t_data}_bw/{fla}_{j}_{tf}',data=t[:,:,:,:1,:4], dtype='complex64')
                                
                os.remove(outfile_flag)
                
    print('flag_cfg_done: '+cfg)
            
run()