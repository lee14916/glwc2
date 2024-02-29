path2=/onyx/qdata/khadjiyiannakou/cC211/disconnected/light_loops_Defl/
for cfg in `cat cfgs`
do
    if ! [ -d "${path2}${cfg}" ]
    then
        echo ${cfg} - not
        continue
    fi

    echo ${cfg}
    mkdir -p data_pre/${cfg}
    for file in ${path2}${cfg}/*exact*gen.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/j.h5_exact_gen
    done
    for file in ${path2}${cfg}/*exact*std.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/j.h5_exact_std
    done
    for file in ${path2}${cfg}/stoch*gen.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/j.h5_stoch_gen
    done
    for file in ${path2}${cfg}/*stoch*std.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/j.h5_stoch_std
    done

    for ind in {1..4}
    do
        for file in ${path2}../strange_loops_S${ind}/${cfg}/*gen.h5
        do
            # echo ${file}
            ln -s ${file} data_pre/${cfg}/js.h5_S${ind}_gen
        done
        for file in ${path2}../strange_loops_S${ind}/${cfg}/*std.h5
        do
            # echo ${file}
            ln -s ${file} data_pre/${cfg}/js.h5_S${ind}_std
        done
    done

    for ind in {1..1}
    do
        for file in ${path2}../charm_loops_S${ind}/${cfg}/*gen.h5
        do
            # echo ${file}
            ln -s ${file} data_pre/${cfg}/jc.h5_S${ind}_gen
        done
        for file in ${path2}../charm_loops_S${ind}/${cfg}/*std.h5
        do
            # echo ${file}
            ln -s ${file} data_pre/${cfg}/jc.h5_S${ind}_std
        done
    done

    # break
done