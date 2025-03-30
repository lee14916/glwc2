for cfg in `cat data_aux/cfgs_run`
do
    echo ${cfg}
    

    inpath=/capstor/store/cscs/userlab/s1174/lyan/code/projectData/NST_f/cA2.09.48_Nsgima_mom0base/data_post/${cfg}/
    outpath=data_post/${cfg}/
    mkdir -p ${outpath}
    for file in `ls ${inpath}`
    do
        echo ${file}
        ln -s ${inpath}${file} ${outpath}${file}
    done

    # break
done