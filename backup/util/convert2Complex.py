import click, h5py
import numpy as np

def complexConverting(t):
    assert(t.shape[-1]==2)
    return t[...,0]+1j*t[...,1]

@click.command()
@click.option('-o','--outfile')
@click.option('-i','--infile')
def run(outfile,infile):
        with h5py.File(infile) as fr, h5py.File(outfile,'w') as fw:
            def visitor_func(name, node):
                if not isinstance(node, h5py.Dataset):
                    return
                if name.endswith('mvec'):
                    fw.copy(node,fw,name=name)
                    return
                fw.create_dataset(name=name,data=complexConverting(node))
            fr.visititems(visitor_func)