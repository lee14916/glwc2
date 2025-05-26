'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u post2avgsrc.py -c @ > log/post2avgsrc.out & 
'''
import os, click, h5py, re
import numpy as np

pathBase='/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/03_NpiScatteringWilson/A15/'

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'{pathBase}data_post/'+cfg+'/'
    outpath=f'{pathBase}data_avgsrc/'+cfg+'/'
    os.makedirs(outpath,exist_ok=True)
    files=os.listdir(inpath)
    
    outfile=outpath+'NPBWZM.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw, h5py.File(f'{inpath}N.h5_0mom') as fN, h5py.File(f'{inpath}P.h5_0mom') as fP, h5py.File(f'{inpath}BWZ.h5_0mom') as fBWZ:
            t=np.mean([fN['data'][src]['N_a'] for src in fN['data'].keys()],axis=0)
            fw.create_dataset('data/N',data=t)
            fw.create_dataset('momsN',data=fN['moms'][:])
            
            t=np.mean([fP['data'][src]['a_a'] for src in fP['data'].keys()],axis=0)
            fw.create_dataset('data/P',data=t)
            t1=np.mean([fP['data'][src]['p+a_p+a'] for src in fP['data'].keys()],axis=0)
            t2=np.mean([fP['data'][src]['p-a_p-a'] for src in fP['data'].keys()],axis=0)
            t=np.concatenate([t1,t2],axis=0)
            fw.create_dataset('data/P_hybrid',data=t)
            fw.create_dataset('momsP',data=fP['moms'][:])
            
            src=list(fBWZ['data'].keys())[0]
            flas=fBWZ[f'data/{src}'].keys()
            for fla in flas:
                t=np.mean([fBWZ['data'][src][fla] for src in fBWZ['data'].keys()],axis=0)
                fw.create_dataset(f'data/{fla}',data=t)
            fw.create_dataset('momsBWZ',data=fBWZ['moms'][:])
            
            t=[]
            imom=16; 
            assert(np.all(fN['moms'][imom]==[0,0,0]))
            for src in fN['data'].keys():
                (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
                (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
                src_new='st{:03d}'.format(st)
                tN=fN[f'data/{src}/N_a'][:,imom,:]
                tP=fP[f'data/{src_new}/a_a'][:,0]
                tM=tN*tP[:,None]
                tM=tM[:,None,:,None]
                t.append(tM)
            t=np.mean(t,axis=0)
            fw.create_dataset(f'data/M',data=t)

        os.remove(outfile_flag)
    print('flag_cfg_done: '+cfg)
    
run()