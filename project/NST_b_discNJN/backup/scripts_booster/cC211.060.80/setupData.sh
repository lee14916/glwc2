# path=/p/arch/hch02/hch02b/cC211.60.80/
# for cfg in `ls ${path}`
# do
#     if [ ${#cfg} != 7 ] || ! [ ${cfg:4:2} == '_r' ]
#     then
#         continue
#     fi
#     echo ${cfg}
#     mkdir -p data_pre/${cfg}
#     for file in `ls ${path}${cfg}/twop_nucl_srcs650.h5`
#     do
#         # echo ${file}
#         ln -s ${file} data_pre/${cfg}/N.h5_twop_nucl_srcs650
#     done
#     # break
# done


path2=/p/project/pines/li47/code/projectData/discNJN/temp/cC211.060.80/
for cfg in `cat cfgs`
do
    echo ${cfg}
    file=${path2}${cfg}/j.h5
    rm data_post/${cfg}/j.h5
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