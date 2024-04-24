for cfg in `cat data_aux/cfgs_run`
do
    echo ${cfg}
    inpath=/project/s1174/lyan/code/projectData/NST_d/cA211.530.24_NJNpi_post/data_post/${cfg}
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