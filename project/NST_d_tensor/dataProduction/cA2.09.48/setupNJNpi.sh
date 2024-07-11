for cfg in `cat data_aux/cfgs_run`
do
    echo ${cfg}
    inpath=/p/project/pines/li47/code/projectData/NST_d-tensor/cA2.09.48_NJNpi_post/data_post/${cfg}
    outpath=data_post/${cfg}
    
    for file in `ls ${inpath}`
    do
        # echo ${file}
        infile=${inpath}/${file}
        outfile=${outpath}/${file}
        ln -s ${infile} ${outfile}
        # break
    done
    # break
done