'''
outfile = infile * factor

bash script template:

for cfg in `ls data_post`
do
    for infile in data_post/${cfg}/*toBeFixed
    do
        echo ${infile}
        outfile=${infile::-10}
        echo ${outfile}
        python3 rescaleData.py -i ${infile} -o ${outfile}
        # break
    done
    echo " "
    # break
done

echo Done!
'''


import h5py, click, shutil, os


flag_remove=False

factor=1/12

def dataQ(name):
    return not name.endswith('mvec')

@click.command()
@click.option('-o','--outfile')
@click.option('-i','--infile')
def run(outfile,infile):
    assert(outfile!=infile)
    assert(not os.path.isfile(outfile))

    with h5py.File(infile) as fi, h5py.File(outfile,'w') as fo:
        def visitor_func(name, node):
            if isinstance(node, h5py.Dataset):
                if dataQ(name):
                    fo.create_dataset(name,data=node[()]*factor)
                else:
                    fo.copy(node,fo,name=name)
                                
        fi.visititems(visitor_func)
    if flag_remove:
        os.remove(infile)
 
if __name__ == '__main__':
    run()