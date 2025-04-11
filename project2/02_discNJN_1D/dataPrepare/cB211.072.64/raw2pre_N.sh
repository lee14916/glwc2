path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cB211.072.64/data_pre_N/

path=/p/arch1/hch02/hch02b/cB211.072.64/twop_threep_dt20_64srcs/
for file in ${path}twop_*.h5
do
    # file_size=$(stat -c %s "$file")
    # if [ ${file_size:0:5} != 38493 ]
    # then
    #     echo ${file}
    # fi
    # echo ${file_size}

    cfg=${file:64:7}
    echo ${cfg}
    mkdir -p ${path_pre}/${cfg}
    ln -s ${file} ${path_pre}/${cfg}/N.h5_twop_threep_dt20_64srcs
    # break
done

path=/p/arch1/hch02/hch02b/cB211.072.64/twop_threep_2/
for file in ${path}twop_*.h5
do
    # file_size=$(stat -c %s "$file")
    # if [ ${file_size:0:5} != 38493 ]
    # then
    #     echo ${file}
    # fi
    # echo ${file_size}

    cfg=${file:54:7}
    echo ${cfg}
    mkdir -p ${path_pre}/${cfg}
    ln -s ${file} ${path_pre}/${cfg}/N.h5_twop_threep_2
    # break
done

path=/p/arch1/hch02/hch02k/cB2.072.64_Nf211/twop/
for file in ${path}*.tar
do
    # file_size=$(stat -c %s "$file")
    # if [ ${file_size:0:5} != 38493 ]
    # then
    #     echo ${file}
    # fi
    # echo ${file_size}

    cfg=${file:44:7}
    echo ${cfg}
    mkdir -p ${path_pre}/${cfg}
    ln -s ${file} ${path_pre}/${cfg}/N.h5_hch02k_twop
    # break
done

