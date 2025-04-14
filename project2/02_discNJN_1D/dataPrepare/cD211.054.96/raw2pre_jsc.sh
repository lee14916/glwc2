# booster
path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cD211.054.96/data_pre_jsc/

for ind in {1..4}
do
    path=/p/arch1/hch02/hch02b/cD211.54.96/strange_S${ind}/
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
                ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/js.h5_S${ind}_gen
            elif [[ ${file} == *std.h5 ]]
            then
                ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/js.h5_S${ind}_std
            else
                echo NONE ${path}${cfg}/${file}
            fi
        done
    done
done

for ind in {1..1}
do
    path=/p/arch1/hch02/hch02b/cD211.54.96/charm_S${ind}/
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
                ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/jc.h5_S${ind}_gen
            elif [[ ${file} == *std.h5 ]]
            then
                ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/jc.h5_S${ind}_std
            else
                echo NONE ${path}${cfg}/${file}
            fi
        done
    done
done