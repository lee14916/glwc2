# path=/p/arch/hch02/hch02b/cD211.54.96/twop_threep_3/
# for file in ${path}twop*.h5
# do
#     cfg=${file:52:7}
#     echo ${cfg}
#     mkdir -p data_pre/${cfg}
#     ln -s ${file} data_pre/${cfg}/N.h5_twop_threep_3
# done


# for t_cfg in `ls ${path}`
# do
#     cfg=${t_cfg:0:7}
#     echo ${cfg}
#     mkdir -p data_pre/${cfg}
#     for file in `ls ${path}${cfg}.h5`
#     do
#         ln -s ${file} data_pre/${cfg}/N.h5_twop_threep_2
#     done
#     # break
# done


path2=/p/project/pines/li47/code/projectData/discNJN/temp/cD211.054.96/
for cfg in `cat cfgs`
do
    echo ${cfg}
    file=${path2}${cfg}/j.h5
    ln -s ${file} data_post/${cfg}/j.h5
    # break
done


# for cfg in `ls alldata`
# do
#     if ! [ -f "alldata/${cfg}/loop.h5" ]; then
#         echo ${cfg}
#         rm -r alldata/${cfg}
#     fi
# done