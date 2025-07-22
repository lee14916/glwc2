path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cE211.044.112/data_pre_j/

path=/p/arch1/hch02/iona1/E112/strange_loops/src1/
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

    echo ${cfg}
    mkdir -p ${path_pre}/${cfg}

    for file in `ls ${path}${cfg}`
    do
        # echo ${file}
        if [[ ${file} == *stoc*gen.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/js.h5_S1_gen
        elif [[ ${file} == *stoc*std.h5 ]]
        then
            ln -s ${path}${cfg}/${file} ${path_pre}/${cfg}/js.h5_S1_std
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
    break
done
