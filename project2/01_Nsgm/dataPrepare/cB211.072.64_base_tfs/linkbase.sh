for line in `cat data_aux/tfs`
do
    tfs=${line}
    break
done

for cfg in `cat ../cB211.072.64_base/data_aux/cfgs_run`
do
    echo ${cfg}
    
    inpath=/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/01_Nsgm/cB211.072.64_base/data_post/${cfg}/
    outpath=/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/01_Nsgm/cB211.072.64_base_${tfs}/data_post/${cfg}/
    mkdir -p ${outpath}
    for file in `ls ${inpath}`
    do
        echo ${file}
        ln -s ${inpath}${file} ${outpath}${file}
    done

    # break
done