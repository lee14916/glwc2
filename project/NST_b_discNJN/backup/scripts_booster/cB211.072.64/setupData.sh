# path=/p/arch/hch02/hch02b/cB211.072.64/twop_threep_dt20_64srcs/
# for file in ${path}twop_*.h5
# do
#     # file_size=$(stat -c %s "$file")
#     # if [ ${file_size:0:5} != 38493 ]
#     # then
#     #     echo ${file}
#     # fi
#     # echo ${file_size}

#     cfg=${file:63:7}
#     echo ${cfg}
#     mkdir -p data_pre/${cfg}
#     ln -s ${file} data_pre/${cfg}/N.h5_twop_threep_dt20_64srcs
#     # break
# done

# path=/p/arch/hch02/hch02b/cB211.072.64/twop_threep_2/
# for file in ${path}twop_*.h5
# do
#     # file_size=$(stat -c %s "$file")
#     # if [ ${file_size:0:5} != 38493 ]
#     # then
#     #     echo ${file}
#     # fi
#     # echo ${file_size}

#     cfg=${file:53:7}
#     echo ${cfg}
#     mkdir -p data_pre/${cfg}
#     ln -s ${file} data_pre/${cfg}/N.h5_twop_threep_2
#     # break
# done

# path=/p/arch/hch02/hch02k/cB2.072.64_Nf211/twop/
# for file in ${path}*.tar
# do
#     # file_size=$(stat -c %s "$file")
#     # if [ ${file_size:0:5} != 38493 ]
#     # then
#     #     echo ${file}
#     # fi
#     # echo ${file_size}

#     cfg=${file:43:7}
#     echo ${cfg}
#     mkdir -p data_pre/${cfg}
#     ln -s ${file} data_pre/${cfg}/N.h5_hch02k_twop
#     # break
# done


path2=/p/project/pines/li47/code/projectData/discNJN/temp/cB211.072.64/
for cfg in `cat cfgs`
do
    echo ${cfg}
    file=${path2}${cfg}/j.h5
    rm data_post/${cfg}/j.h5
    ln -s ${file} data_post/${cfg}/j.h5
    # break
done

# echo remove

# for cfg in `ls data_pre`
# do
#     if ! [ -f "data_pre/${cfg}/jLoop.h5" ]; then
#         echo ${cfg}
#         rm -r data_pre/${cfg}
#     fi
# done