# cyclone
path_pre=/nvme/h/cy22yl1/projectData/02_discNJN_1D/cC211.060.80/data_pre_j/

path=/onyx/qdata/khadjiyiannakou/cC211/disconnected/light_loops_Defl/
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
        if [[ ${file} == *exact*gen.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_exact_gen
        elif [[ ${file} == *exact*std.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_exact_std
        elif [[ ${file} == stoch*gen.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_stoch_gen
        elif [[ ${file} == *stoch*std.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_stoch_std
        else
            echo NONE ${path}${cfg}/${file}
        fi
    done
done

for ind in {1..4}
do
    path=/onyx/qdata/khadjiyiannakou/cC211/disconnected/strange_loops_S${ind}/
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
    path=/onyx/qdata/khadjiyiannakou/cC211/disconnected/charm_loops_S${ind}/
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