for cfg in `cat data_aux/cfgs_run`
do
    echo ${cfg}

    inpath=/project/s1174/lyan/code/projectData/fromBooster/projectDataMigrating/NST_c/cA2.09.48/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]] || [[ ${file} == NJN.h5* ]] ;
        then
            if [ ! -h ${inpath}${file} ]
            then
                # echo ${file}
                ln -s ${inpath}${file} ${outpath}${file}
            fi
        fi
    done


    inpath=/project/s1174/lyan/code/projectData/fromBooster/projectDataMigrating/NST_c/cA2.09.48_NJNpi_post/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]] ;
        then
            if [ ! -h ${inpath}${file} ]
            then
                # echo ${file}
                ln -s ${inpath}${file} ${outpath}${file}
            fi
        fi
    done

    inpath=/project/s1174/lyan/code/projectData/fromBooster/projectDataMigrating/NST_d-tensor/cA2.09.48/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]] || [[ ${file} == NJN.h5* ]] ;
        then
            if [ ! -h ${inpath}${file} ]
            then
                # echo ${file}
                ln -s ${inpath}${file} ${outpath}${file}
            fi
        fi
    done

    inpath=/project/s1174/lyan/code/projectData/fromBooster/projectDataMigrating/NST_d-tensor/cA2.09.48_NJNpi_post/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]] ;
        then
            if [ ! -h ${inpath}${file} ]
            then
                # echo ${file}
                ln -s ${inpath}${file} ${outpath}${file}
            fi
        fi
    done

    inpath=/project/s1174/lyan/code/projectData/NST_f/cA2.09.48_Nsigma_NJNpi/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        if [[ ${file} == *.h5_NJNpi-f-Nsrc4*1-Nsigma ]];
        then
            # echo ${file}
            ln -s ${inpath}${file} ${outpath}${file}
        fi
    done
    
    # break
done