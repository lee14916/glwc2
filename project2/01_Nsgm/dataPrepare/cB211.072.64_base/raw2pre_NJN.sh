path_pre=/p/project/ngff/li47/code/projectData2/01_Nsgm/cB211.072.64_base/data_pre_NJN/

path=/p/arch1/hch02/hch02b/cB211.072.64/twop_threep_dt20_64srcs/
for file in ${path}threep_*.h5
do
    # file_size=$(stat -c %s "$file")
    # if [ ${file_size:0:5} != 38493 ]
    # then
    #     echo ${file}
    # fi
    # echo ${file_size}

    cfg=${file:66:7}
    echo ${cfg}
    mkdir -p ${path_pre}/${cfg}
    ln -s ${file} ${path_pre}/${cfg}/NJN.h5_twop_threep_dt20_64srcs
    # break
done

path=/p/arch1/hch02/hch02b/cB211.072.64/twop_threep_2/
for file in ${path}threep_*.h5
do
    # file_size=$(stat -c %s "$file")
    # if [ ${file_size:0:5} != 38493 ]
    # then
    #     echo ${file}
    # fi
    # echo ${file_size}

    cfg=${file:56:7}
    echo ${cfg}
    mkdir -p ${path_pre}/${cfg}
    ln -s ${file} ${path_pre}/${cfg}/NJN.h5_twop_threep_2
    # break
done