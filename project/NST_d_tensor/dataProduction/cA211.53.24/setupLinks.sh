for cfg in `cat data_aux/cfgs_run`
do
    echo ${cfg}
    inpath=/project/s1174/lyan/code/projectData/NST_a/cA211.530.24/data_post/${cfg}/
    outpath=data_post/${cfg}/
    for file in `ls ${inpath}`
    do
        # W2pt_bw.h5_NJNpi-a-Nsrc3*3 is missing in NST_a, so I give up all BWZ there
        if [[ ${file} == N.h5* ]] || [[ ${file} == N_bw.h5* ]] || [[ ${file} == T*.h5* ]];
        then
            echo ${file}
            ln -s ${inpath}${file} ${outpath}${file}
        fi
    done
    # break
done