for cfg in `cat data_aux/cfgs_run`
do
    # echo ${cfg}
    # inpath=/project/s1174/lyan/code/projectData/NST_a/cA211.530.24/data_post/${cfg}/
    # outpath=data_post/${cfg}/
    # for file in `ls ${inpath}`
    # do
    #     if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]];
    #     then
    #         echo ${file}
    #         ln -s ${inpath}${file} ${outpath}${file}
    #     fi
    # done

    # inpath=/project/s1174/lyan/code/projectData/NST_d/cA211.530.24_NJNpi_post/data_post/${cfg}/
    # outpath=data_post/${cfg}/
    # for file in `ls ${inpath}`
    # do
    #     if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]];
    #     then
    #         echo ${file}
    #         ln -s ${inpath}${file} ${outpath}${file}
    #     fi
    # done

    # inpath=/project/s1174/lyan/code/projectData/NST_d/cA211.530.24/data_post/${cfg}/
    # outpath=data_post/${cfg}/
    # for file in `ls ${inpath}`
    # do
    #     if [[ ${file} == *.h5_NJN-b-Nsrc1*4-tensor ]];
    #     then
    #         echo ${file}
    #         ln -s ${inpath}${file} ${outpath}${file}
    #     fi
    # done

    inpath=/project/s1174/lyan/code/projectData/NST_f/cA211.53.24_Nsigma_NJNpi/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        if [[ ${file} == *.h5_NJNpi-f-Nsrc3*4-Nsigma ]];
        then
            echo ${file}
            ln -s ${inpath}${file} ${outpath}${file}
        fi
    done
    
    # break
done