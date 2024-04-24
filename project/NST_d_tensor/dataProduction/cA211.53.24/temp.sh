# for cfg in `ls data_post`
# do  
#     echo ${cfg}
#     for file in `ls data_post/${cfg}/*.h5_N-a-Nsrc4*32`
#     do
#         echo ${file}
#         rm ${file}
#     done
#     # break
# done


for cfg in `ls data_avgsrc`
do  
    echo ${cfg}
    for file in `ls data_avgsrc/${cfg}/N_bw.h5`
    do
        echo ${file}
        rm ${file}
    done
    # break
done