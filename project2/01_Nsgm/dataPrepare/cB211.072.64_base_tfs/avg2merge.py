'''
rm -r data_merge/temp*
cat data_aux/diags_all | xargs -I @ -P 10 python3 -u avg2merge.py -d @ > log/avg2merge.out & 
cat data_aux/diags_main | xargs -I @ -P 10 python3 -u avg2merge.py -d @ > log/avg2merge.out & 
'''
import os, click, h5py, re
import numpy as np
import auxx as aux

inpath=f'{aux.pathBaseTf}data_avgsrc/'
inpath=f'{aux.pathBaseTf}data_avgmore/'

@click.command()
@click.option('-d','--diag')
def run(diag):
    outpath=f'{aux.pathBaseTf}data_merge/temp1/'
    
    with open(aux.path_cfgs) as f:
        cfgs=f.read().splitlines()
    
    outfile=outpath+diag+'.h5'
    os.makedirs(outpath,exist_ok=True)
    assert(not os.path.isfile(outfile))
    with h5py.File(outfile,'w') as fw:
        for cfg in cfgs:
            infile=inpath+cfg+'/'+diag+'.h5'
            with h5py.File(infile) as fr:
                for ky in ['opabs','inserts']:
                    if ky not in fw.keys() and ky in fr.keys():
                        fw.copy(fr[ky],fw,name=ky)
                for ky in ['srcs','data']:
                    fw.copy(fr[ky],fw,name=ky+'/'+cfg)
                        
    # print('flag_diag_done: '+diag)
    
    inpath2=f'{aux.pathBaseTf}data_merge/temp1/'
    outpath2=f'{aux.pathBaseTf}data_merge/temp2/'
    
    outfile=outpath2+diag+'.h5'
    os.makedirs(outpath2,exist_ok=True)
    with h5py.File(outfile,'w') as fw:
        infile=inpath2+diag+'.h5'
        with h5py.File(infile) as fr:
            for ky in ['opabs','inserts']:
                if ky in fr.keys():
                    fw.copy(fr[ky],fw,name=ky)
            fw.create_dataset('cfgs',data=cfgs)
            for ky in ['srcs','data']:
                for ky2 in fr[ky][cfgs[0]].keys():
                    t=np.array([fr[ky][cfg][ky2][:] for cfg in cfgs])
                    fw.create_dataset(ky+'/'+ky2,data=t)
                        
    print('flag_diag_done: '+diag)
                
run()