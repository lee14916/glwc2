# cyclone run
path=/onyx/qdata/khadjiyiannakou/cB2.072.64_Nf211/disconnected/light_loops/
path_pre=/nvme/h/cy22yl1/projectData/02_discNJN_1D/cB211.072.64/data_pre_j/
for cfg in `ls ${path}`
do
    if ! [[ ${cfg} =~ ^[0-9]{4}_r[0-9]$ ]]
    then
        echo NONE0 ${path}${cfg}
        continue
    fi

    if ! [ -d "${path}${cfg}" ]
    then
        continue
    fi

    # echo ${cfg}
    mkdir -p ${path_pre}/${cfg}

    for file in `ls ${path}${cfg}`
    do
        # echo ${file}
        if [[ ${file} == *exact*.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_exact
        elif [[ ${file} == *stoch*.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/j.h5_stoch
        else
            echo NONE ${path}${cfg}/${file}
        fi
    done
    # break
done

path=/onyx/qdata/khadjiyiannakou/cB2.072.64_Nf211/disconnected/strange_loops_D8/
for cfg in `ls ${path}`
do
    if ! [[ ${cfg} =~ ^[0-9]{4}_r[0-9]$ ]]
    then
        echo NONE0 ${path}${cfg}
        continue
    fi

    if ! [ -d "${path}${cfg}" ]
    then
        continue
    fi

    # echo ${cfg}
    mkdir -p ${path_pre}/${cfg}

    for file in `ls ${path}${cfg}`
    do
        # echo ${file}
        if [[ ${file} == *stoc*NeV0*.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/js.h5_stoch_D8
        else
            echo NONE ${path}${cfg}/${file}
        fi
    done
    # break
done

path=/onyx/qdata/khadjiyiannakou/cB2.072.64_Nf211/disconnected/strange_loops_D8_S2/
for cfg in `ls ${path}`
do
    if ! [[ ${cfg} =~ ^[0-9]{4}_r[0-9]$ ]]
    then
        echo NONE0 ${path}${cfg}
        continue
    fi

    if ! [ -d "${path}${cfg}" ]
    then
        continue
    fi

    # echo ${cfg}
    mkdir -p ${path_pre}/${cfg}

    for file in `ls ${path}${cfg}`
    do
        # echo ${file}
        if [[ ${file} == *stoc*gen.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/js.h5_stoch_gen_D8_S2
        elif [[ ${file} == *stoc*std.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/js.h5_stoch_std_D8_S2
        elif [[ ${file} == *.out ]]
        then
            continue
        elif [[ ${file} == *.err ]]
        then
            continue
        else
            echo NONE ${path}${cfg}/${file}
        fi
    done
    # break
done

path=/onyx/qdata/khadjiyiannakou/cB2.072.64_Nf211/disconnected/charm_loops/
for cfg in `ls ${path}`
do
    if ! [[ ${cfg} =~ ^[0-9]{4}_r[0-9]$ ]]
    then
        echo NONE0 ${path}${cfg}
        continue
    fi

    if ! [ -d "${path}${cfg}" ]
    then
        continue
    fi

    # echo ${cfg}
    mkdir -p ${path_pre}/${cfg}

    for file in `ls ${path}${cfg}`
    do
        # echo ${file}
        if [[ ${file} == *stoc*.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/jc.h5_stoch
        else
            echo NONE ${path}${cfg}/${file}
        fi
    done
    # break
done