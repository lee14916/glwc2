# for cfg in `ls data_post`
# do  
#     echo ${cfg}
#     for file in `ls data_post/${cfg}/j.h5_loops-a-Nstoc400`
#     do
#         echo ${file}
#         rm ${file}
#     done
#     for file in `ls data_post/${cfg}/jPi.h5_jPP-a-tensor`
#     do
#         echo ${file}
#         rm ${file}
#     done
#     # break
# done


# for cfg in `ls data_avgsrc`
# do  
#     echo ${cfg}
#     for file in `ls data_avgsrc/${cfg}/*-j*.h5`
#     do
#         echo ${file}
#         rm ${file}
#     done
#     # break
# done

for file in `ls data_avgsrc/a0240`
do
    echo ${file}
    h5diff data_avgsrc/a0240/${file} data_avgsrc_backup2/a0240/${file} 
    # echo ''
done