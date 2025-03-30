tfs=8

for cfg in `cat /capstor/store/cscs/userlab/s1174/lyan/code/glwc2/project2/Nsgm/dataPrepare/cB211.072.64_base/data_aux/cfgs_run`
do
    echo ${cfg}
    
    inpath=/capstor/store/cscs/userlab/s1174/lyan/code/projectData/Nsgm/cB211.072.64_base/data_post/${cfg}/
    outpath=/capstor/store/cscs/userlab/s1174/lyan/code/projectData/Nsgm/cB211.072.64_base_${tfs}/data_post/${cfg}/
    mkdir -p ${outpath}
    for file in `ls ${inpath}`
    do
        echo ${file}
        ln -s ${inpath}${file} ${outpath}${file}
    done

    # break
done