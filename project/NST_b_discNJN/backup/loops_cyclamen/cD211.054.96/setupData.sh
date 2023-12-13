path2=/onyx/qdata/khadjiyiannakou/cD211/disconnected/
for cfg in `cat cfgs`
do
    for ind in {1..8}
    do
        if ! [ -d "${path2}light_S${ind}" ]
        then
            echo ${ind}: ${cfg} - not
        fi
    done

    echo ${cfg}
    mkdir -p data_pre/${cfg}
    for ind in {1..8}
    do
        for file in ${path2}light_S${ind}/${cfg}/*gen.h5
        do
            echo ${file}
            ln -s ${file} data_pre/${cfg}/j.h5_S${ind}_gen
        done
        for file in ${path2}light_S${ind}/${cfg}/*std.h5
        do
            echo ${file}
            ln -s ${file} data_pre/${cfg}/j.h5_S${ind}_std
        done
    done
    # break
done