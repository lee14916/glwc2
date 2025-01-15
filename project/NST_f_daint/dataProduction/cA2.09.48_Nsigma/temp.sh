# for cfg in `ls data_post`
# do  
#     echo ${cfg}
#     for file in `ls data_post/${cfg}/j.h5_loops-b-Nstoc200-tensor`
#     do
#         echo ${file}
#         rm ${file}
#     done
#     for file in `ls data_post/${cfg}/jPi.h5_jPP-b-tensor`
#     do
#         echo ${file}
#         rm ${file}
#     done
#     # break
# done


for cfg in `ls data_avgsrc`
do  
    echo ${cfg}
    for file in `ls data_avgsrc/${cfg}/*-j*.h5`
    do
        echo ${file}
        rm ${file}
    done
    # break
done