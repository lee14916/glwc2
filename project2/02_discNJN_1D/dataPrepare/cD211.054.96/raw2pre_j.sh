# cyclone
path_pre=/nvme/h/cy22yl1/projectData/02_discNJN_1D/cD211.054.96/data_pre_j/

for ind in {1..8}
do
    path=/onyx/qdata/khadjiyiannakou/cD211/disconnected/light_S${ind}/
    for cfg in `ls ${path}`
    do
        if ! [[ ${cfg} =~ ^[0-9]{4}_r[0-9]$ ]]
        then
            echo NONE0 ${path}${cfg}
            continue
        fi

        # echo ${cfg}
        mkdir -p ${path_pre}${cfg}

        for file in `ls ${path}${cfg}`
        do
            if [[ ${file} == *gen.h5 ]]
            then
                ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_S${ind}_gen
            elif [[ ${file} == *std.h5 ]]
            then
                ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_S${ind}_std
            else
                echo NONE ${path}${cfg}/${file}
            fi
        done
    done
done