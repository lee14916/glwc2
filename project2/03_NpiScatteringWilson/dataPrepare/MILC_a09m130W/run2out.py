'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u run2out.py -c @ > log/run2out.out & 
'''
import os, click, h5py, re, pickle
import numpy as np

runPath='/capstor/store/cscs/userlab/lp139/lyan/code/projectData/03_NpiScatteringWilson/MILC_a09m130W/'

lat_L,lat_T=64,96

moms_target=[[0,0,0, 0,0,0, 0,0,0, 0,0,0]] # pi2,pf1,pf2,pc=000

type2base={
    '11':'gc,gd,abdc->ab',
    '12':'dc,gd,bagc->ab',
    '13':'bc,gd,dagc->ab',
    '21':'bc,gd,agdc->ab',
    '22':'dc,gd,gabc->ab',
    '23':'gc,gd,dabc->ab',
}
def type2pat_V3V24(tp,extra_V3,extra_V24,extra_V3V24):
    t=type2base[tp]
    t1,t2=t.split('->')
    t11,t12,t13=t1.split(',')
    return f'{t11}{extra_V3},{t12},{t13}{extra_V24}->{t2}{extra_V3V24}' 

def key2mom(mom_key):
    t1,t2=mom_key.split('=')
    x,y,z=t2.split('_')
    x,y,z=int(x),int(y),int(z)
    return [x,y,z]
def moms2dic(moms):
    dic={}
    for i,mom in enumerate(moms):
        dic[tuple(mom)]=i
    return dic

Format_ab=4
a2ab_map=[a for a in range(Format_ab) for b in range(Format_ab)]
b2ab_map=[b for a in range(Format_ab) for b in range(Format_ab)]

gamma_1=gamma_x=np.array([[0.,0.,0.,1j],[0.,0.,1j,0.],[0.,-1j,0.,0.],[-1j,0.,0.,0.]])
gamma_2=gamma_y=np.array([[0.,0.,0.,1.],[0.,0.,-1.,0.],[0.,-1.,0.,0.],[1.,0.,0.,0.]])
gamma_3=gamma_z=np.array([[0.,0.,1j,0.],[0.,0.,0.,-1j],[-1j,0.,0.,0.],[0.,1j,0.,0.]])
gamma_4=gamma_t=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,-1.]])
gamma_5=(gamma_1@gamma_2@gamma_3@gamma_4)

cg5=1j*gamma_2@gamma_4@gamma_5
Gm_i1=cg5

def get_phase(src_int,mom):
    (sx,sy,sz,st)=src_int
    return np.exp(1j*(2*np.pi/lat_L)*(np.array([sx,sy,sz])@mom))
def mom2pi1(mom):
    pi2=mom[0:3]; pf1=mom[3:6]; pf2=mom[6:9]; pc=mom[9:12]
    pi1=np.array(pf1)+pf2+pc-pi2
    return pi1

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'{runPath}run/{cfg}/'
    outpath=f'{runPath}out/{cfg}/'
    os.makedirs(outpath,exist_ok=True)
    
    files=[file for file in os.listdir(inpath) if file.endswith('.h5')]
    
    for file in files:
        if file.endswith('N.h5'):
            infile=f'{inpath}{file}'
            outfile=f'{outpath}{file}'
            outfile_flag=outfile+'_flag'
            if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
                with open(outfile_flag,'w') as f:
                    pass
                with h5py.File(outfile,'w') as fw, h5py.File(infile) as f:
                    for key in f.keys():
                        fw.copy(f[key],fw,name=key)
                os.remove(outfile_flag)
        
        if file.endswith('NpiScatteringWilson.h5'):
            infile=f'{inpath}{file}'
            outfile=f'{outpath}{file}'
            outfile_flag=outfile+'_flag'
            if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
                with open(outfile_flag,'w') as f:
                    pass
                with h5py.File(outfile,'w') as fw, h5py.File(infile) as f:
                    src=list(f.keys())[0]
                    (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',file).groups()
                    (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                    src_new='sx{:03d}sy{:03d}sz{:03d}st{:03d}'.format(sx,sy,sz,st)
                    Nstoc=len([stoc for stoc in f[f'{src}/V2B_2'].keys() if stoc.startswith('i_stoc=')])
                    
                    pi2s_key=[pi2 for pi2 in f[f'{src}/V3B_2'].keys() if pi2.startswith('pi2=')]; pi2s_key.sort()
                    pi2s=[key2mom(pi2) for pi2 in pi2s_key]
                    pf1s=[list(mom) for mom in np.atleast_2d(f[f'{src}/V2B_2/mvec'])]
                    pf2s=[list(mom) for mom in np.atleast_2d(f[f'{src}/V3B_2/mvec'])]
                    
                    pi2Map=[pi2s.index(mom[:3]) for mom in moms_target]
                    pf1Map=[pf1s.index(mom[3:6]) for mom in moms_target]
                    pf2Map=[pf2s.index(mom[6:9]) for mom in moms_target]
                    
                    phases=np.array([get_phase((sx,sy,sz,st),mom2pi1(mom)) for mom in moms_target])
                    
                    # print(f[f'{src}/V2B1/i_stoc=0'])
                    # print(f[f'{src}/V3B1/pi2=0_0_0/i_stoc=0'])
                    # print(f[f'{src}/V4B1/i_stoc=0'])
                    
                    # print(f[f'{src}/V2W1a/pi2=0_0_0/i_stoc=0'])
                    # print(f[f'{src}/V3W1/i_stoc=0'])
                    # print(f[f'{src}/V4W1a/pi2=0_0_0/i_stoc=0'])
                    
                    notes=['time,mom,dirac,stoc(,stoc2)','mom=[pi2,pf1,pf2]','dirac=4a+b']
                    fw.create_dataset('notes',data=notes)
                    fw.create_dataset('moms',data=moms_target)
                    
                    conts=[
                        # 'B114_1','B122_1','B132_1','B214_1','B232_1',
                           'B114_2','B122_2','B132_2','B214_2','B232_2',
                           'Z114_1','Z122_1','Z132_1','Z214_1','Z222_1'
                           ]
                    for cont in conts:
                        topo=cont[0]; tp=cont[1:3]; V24=cont[3]; md=cont[-1]
                        kyV3=f'V3{topo}_{md}'; kyV24=f'V{V24}{topo}_{md}'; kyV3V24=f'{cont}'
                        if topo=='W':
                            seqWhere=cont[4]
                            kyV24=f'V{V24}{topo}{seqWhere}_{md}'
                        t_V3=np.array([[f[f'{src}/{kyV3}/{pi2_key}/i_stoc={istoc}'][:,:,0] for pi2_key in pi2s_key] for istoc in range(Nstoc)])
                        t_V3=t_V3[...,0]+1j*t_V3[...,1]
                        t_V3=np.transpose(t_V3,[4,5,2,3,1,0]) # spin,color, time,xmom=pf2,pi2,stoc
                        t_V24=np.array([f[f'{src}/{kyV24}/i_stoc={istoc}'][:,:,0] for istoc in range(Nstoc)])
                        t_V24=t_V24[...,0]+1j*t_V24[...,1]
                        t_V24=np.transpose(t_V24,[3,4,5,6,1,2,0]) # spin1,spin2,spin3,color, time,ymom=pf1,stoc
                        pat=type2pat_V3V24(tp,'txps','tys','tpyxs')
                        t=np.einsum(pat,t_V3,Gm_i1,t_V24,optimize='optimal') # alpha,beta,time,pi2,pf1,pf2,stoc
                        t=np.transpose(t,[2,3,4,5,0,1,6]) # time,pi2,pf1,pf2,alpha,beta,stoc
                        t=t[:,pi2Map,pf1Map,pf2Map]
                        t=t*phases[None,:]
                        t=t[:,:,a2ab_map,b2ab_map]
                        fw.create_dataset(f'data/{src_new}/{kyV3V24}',data=t.astype('complex64'))
                        
                    conts=[
                        # 'W1141_1','W1142_1','W1221_1','W1222_1','W1321_1','W1322_1','W2321_1','W2142_1','W2322_1',
                           'W1141_2','W1142_2','W1221_2','W1222_2','W1321_2','W1322_2','W2321_2','W2142_2','W2322_2'
                           ]
                    for cont in conts:
                        topo=cont[0]; tp=cont[1:3]; V24=cont[3]; md=cont[-1]
                        kyV3=f'V3{topo}_{md}'; kyV24=f'V{V24}{topo}_{md}'; kyV3V24=f'{cont}'
                        if topo=='W':
                            seqWhere=cont[4]
                            kyV24=f'V{V24}{topo}{seqWhere}_{md}'
                        
                        t_V3=np.array([f[f'{src}/{kyV3}/i_stoc={istoc}'][:,:,0] for istoc in range(Nstoc)])
                        t_V3=t_V3[...,0]+1j*t_V3[...,1]
                        t_V3=np.transpose(t_V3,[3,4,1,2,0]) # spin,color, time,xmom=pf2,stoc
                        t_V24=np.array([[f[f'{src}/{kyV24}/{pi2_key}/i_stoc={istoc}'][:,:,0] for pi2_key in pi2s_key] for istoc in range(Nstoc)])
                        t_V24=t_V24[...,0]+1j*t_V24[...,1]
                        t_V24=np.transpose(t_V24,[4,5,6,7,2,3,1,0]) # spin1,spin2,spin3,color, time,ymom=pf1,pi2,stoc
                        pat=type2pat_V3V24(tp,'txs','typs','tpyxs')
                        t=np.einsum(pat,t_V3,Gm_i1,t_V24,optimize='optimal') # alpha,beta,time,pi2,pf1,pf2,stoc
                        t=np.transpose(t,[2,3,4,5,0,1,6]) # time,pi2,pf1,pf2,alpha,beta,stoc
                        t=t[:,pi2Map,pf1Map,pf2Map]
                        t=t*phases[None,:]
                        t=t[:,:,a2ab_map,b2ab_map]
                        fw.create_dataset(f'data/{src_new}/{kyV3V24}',data=t.astype('complex64'))
                        
                    
                    # 2 stocs (only for all 0 moms)
                    assert(phases.shape==(1,) and phases[0]==1)
                    conts=[
                        # 'B114_4','B122_4','B132_4','B214_4','B232_4',
                        #    'Z114_2','Z122_2','Z132_2','Z214_2','Z222_2'
                           ]
                    for cont in conts:
                        topo=cont[0]; tp=cont[1:3]; V24=cont[3]; md=cont[-1]
                        kyV3=f'V3{topo}_{md}'; kyV24=f'V{V24}{topo}_{md}'; kyV3V24=f'{cont}'
                        if topo=='W':
                            seqWhere=cont[4]
                            kyV24=f'V{V24}{topo}{seqWhere}_{md}'
                            
                        if topo=='B':
                            kyV3=f'V3W_2'; kyV24=f'V{V24}{topo}_{2}'; kyPhiPhi=f'PhiPhiB_4'
                        elif topo=='Z':
                            kyV3=f'V3W_2'; kyV24=f'V{V24}{topo}_{1}'; kyPhiPhi=f'PhiPhiB_4'
                            
                        t_PhiPhi=np.array([[f[f'{src}/{kyPhiPhi}/i_stoc={istoc}/j_stoc={jstoc}'][:,0,0] for jstoc in range(Nstoc)] for istoc in range(Nstoc)])
                        t_PhiPhi=t_PhiPhi[...,0]+1j*t_PhiPhi[...,1]
                        t_PhiPhi=np.transpose(t_PhiPhi,[2,0,1]) # tf,i,j
                        if topo=='Z':
                            t_PhiPhi=np.transpose(np.conjugate(t_PhiPhi),[0,2,1])
                            
                        t_V24=np.array([f[f'{src}/{kyV24}/i_stoc={istoc}'][:,:,0] for istoc in range(Nstoc)])
                        t_V24=t_V24[...,0]+1j*t_V24[...,1]
                        t_V24=np.transpose(t_V24,[3,4,5,6,1,2,0]) # spin1,spin2,spin3,color, time,ymom=pf1,stoc
                        
                        t_V3=np.array([f[f'{src}/{kyV3}/i_stoc={istoc}'][:,:,0] for istoc in range(Nstoc)])
                        t_V3=t_V3[...,0]+1j*t_V3[...,1]
                        t_V3=np.transpose(t_V3,[3,4,1,2,0]) # spin,color, time,xmom=pf2,stoc
                        if topo=='B':
                            t_V3=t_V3[:,:,0] # spin,color, xmom,jstoc
                            t_V3=np.einsum('scxj,tij->sctxij',t_V3,t_PhiPhi) # spin,color, xmom,istoc,jstoc
                        elif topo=='Z':
                            t_V3=np.einsum('sctxj,tij->sctxij',t_V3,t_PhiPhi)
                        
                        t_V3=t_V3[...,None] # add p dimension
                        pat=type2pat_V3V24(tp,'txijp','tyi','tpyxij')
                        t=np.einsum(pat,t_V3,Gm_i1,t_V24,optimize='optimal') # alpha,beta,time,pi2,pf1,pf2,istoc,jstoc
                        t=np.transpose(t,[2,3,4,5,0,1,6,7]) # time,pi2,pf1,pf2,alpha,beta,istoc,jstoc
                        t=t[:,pi2Map,pf1Map,pf2Map]
                        t=t*phases[None,:]
                        t=t[:,:,a2ab_map,b2ab_map]
                        fw.create_dataset(f'data/{src_new}/{kyV3V24}',data=t.astype('complex64'))
                        
                    conts=[
                        # 'W1141_3','W1142_3','W1221_3','W1222_3','W1321_3','W1322_3','W2321_3','W2142_3','W2322_3'
                        ]
                    for cont in conts:
                        topo=cont[0]; tp=cont[1:3]; V24=cont[3]; md=cont[-1]
                        kyV3=f'V3{topo}_{2}'; kyV24=f'V{V24}{topo}_{md}'; kyV3V24=f'{cont}'
                        if topo=='W':
                            seqWhere=cont[4]
                            kyV24=f'V{V24}{topo}{seqWhere}_{md}'
                        
                        t_V3=np.array([f[f'{src}/{kyV3}/i_stoc={istoc}'][:,:,0] for istoc in range(Nstoc)])
                        t_V3=t_V3[...,0]+1j*t_V3[...,1]
                        t_V3=np.transpose(t_V3,[3,4,1,2,0]) # spin,color, time,xmom=pf2,stoc
                        t_V24=np.array([[[f[f'{src}/{kyV24}/i_stoc={istoc}/j_stoc={jstoc}'][:,:,0] for pi2_key in pi2s_key] for jstoc in range(Nstoc)] for istoc in range(Nstoc)])
                        t_V24=t_V24[...,0]+1j*t_V24[...,1]
                        t_V24=np.transpose(t_V24,[5,6,7,8,3,4,2,0,1]) # spin1,spin2,spin3,color, time,ymom=pf1,pi2,i,j
                        pat=type2pat_V3V24(tp,'txi','typij','tpyxij')
                        t=np.einsum(pat,t_V3,Gm_i1,t_V24,optimize='optimal') # alpha,beta,time,pi2,pf1,pf2,stoc
                        t=np.transpose(t,[2,3,4,5,0,1,6,7]) # time,pi2,pf1,pf2,alpha,beta,stoc
                        t=t[:,pi2Map,pf1Map,pf2Map]
                        t=t*phases[None,:]
                        t=t[:,:,a2ab_map,b2ab_map]
                        fw.create_dataset(f'data/{src_new}/{kyV3V24}',data=t.astype('complex64'))

                os.remove(outfile_flag)
        
    print('flag_cfg_done: '+cfg)
    
run()