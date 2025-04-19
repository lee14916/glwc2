'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u post2avgsrc.py -c @ > log/post2avgsrc.out & 
'''
import h5py,os,re,click
import numpy as np

ens='cB211.072.64'

lat_L={'cB211.072.64':64,'cC211.060.80':80,'cD211.054.96':96}[ens]

max_mom2={'cB211.072.64':23,'cC211.060.80':26,'cD211.054.96':26}[ens]
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms_pc=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]
max_mom2=1
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms_pf=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]

moms_target=[pf+pc for pf in moms_pf for pc in moms_pc]
moms_target.sort()
# moms_target=np.array(moms_target)

tfs={'cB211.072.64':range(2,26+1),'cC211.060.80':range(2,28+1),'cD211.054.96':range(2,32+1)}[ens]

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
    inpath=f'/p/project/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post/{cfg}/'
    outpath=f'/p/project/ngff/li47/code/scratch/run/02_discNJN_1D/{ens}/data_avgsrc/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    files=[file for file in os.listdir(inpath) if file.startswith('N.h5')]
    
    inpath_fullmom=f'/p/project/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_N_fullmom/{cfg}/'
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
            for key,val in data.items():
                fla=key
                fw.create_dataset(f'data/{fla}',data=data[fla]/Nsrc)    
                fw.create_dataset(f'data_bw/{fla}',data=data_bw[fla]/Nsrc)    
        os.remove(outfile_flag)
    
    with h5py.File(f'{inpath}j.h5') as fj:
        flas=['N_N']
        # js=['j+','j-','js','jc']
        # js=[j for j in fj['data'].keys() if not j.startswith('j-') and ';' in j]
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
                    if ';' in j:
                        t=j.split(';')[1:]; t=';'.join(t)
                        ky=f'inserts;{t}'
                    else:
                        ky='inserts'
                    fw.create_dataset(ky,data=fj[ky][:])
                    for key in data:
                        fla,tf,j=key
                        fw.create_dataset(f'data/{fla}_{j}_{tf}',data=data[key]/Nsrc)
                        fw.create_dataset(f'data_bw/{fla}_{j}_{tf}',data=data_bw[key]/Nsrc)
                            
                os.remove(outfile_flag)
                
    print('flag_cfg_done: '+cfg)
            
run()