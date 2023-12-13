import h5py, click, shutil

@click.command()
@click.option('-o','--outfile')
@click.option('-b','--basefile')
@click.option('-i','--infile')
def run(outfile,basefile,infile):
    assert(outfile!=basefile)
    assert(outfile!=infile)
    with h5py.File(outfile,'w') as fo, h5py.File(basefile) as fb, h5py.File(infile) as fi:
        fo.copy(fb,fo)
        def visitor_func(name, node):
            if isinstance(node, h5py.Dataset):
                fo[name]=node
                                
        fi.visititems(visitor_func)

 
if __name__ == '__main__':
    run()