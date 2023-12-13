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
    # break
done