path2=/onyx/qdata/khadjiyiannakou/cB2.072.64_Nf211/disconnected/light_loops/
for cfg in `cat cfgs`
do
    if ! [ -d "${path2}${cfg}" ]
    then
        echo ${cfg} - not
        continue
    fi

    echo ${cfg}
    mkdir -p data_pre/${cfg}
    for file in ${path2}${cfg}/*exact*.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/j.h5_exact
    done
    for file in ${path2}${cfg}/*stoc*.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/j.h5_stoch
    done
    for file in ${path2}../strange_loops_D8/${cfg}/*stoc*NeV0*.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/js.h5_stoch_D8
    done
    for file in ${path2}../strange_loops_D8_S2/${cfg}/*stoc*gen.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/js.h5_stoch_gen_D8_S2
    done
    for file in ${path2}../strange_loops_D8_S2/${cfg}/*stoc*std.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/js.h5_stoch_std_D8_S2
    done
    for file in ${path2}../charm_loops/${cfg}/*stoc*.h5;
    do
        # echo ${file}
        ln -s ${file} data_pre/${cfg}/jc.h5_stoch
    done
    # break
done