for cfg in `cat data_aux/cfgs_run`
do
    echo ${cfg}
    inpath=/p/project/pines/li47/code/projectData/NST_c/cA2.09.48/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]] || [[ ${file} == T*.h5* ]] || [[ ${file} == B2pt*.h5* ]] || [[ ${file} == W2pt*.h5* ]] || [[ ${file} == Z2pt*.h5* ]] ;
        then
            echo ${file}
            ln -s ${inpath}${file} ${outpath}${file}
        fi
    done
    # break
done