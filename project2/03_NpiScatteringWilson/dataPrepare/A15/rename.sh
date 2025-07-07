path=/capstor/store/cscs/userlab/s1174/lyan/code/projectData2/03_NpiScatteringWilson/A15/data_post/

# for cfg in `ls ${path}`
# do 
#     echo ${cfg}
#     mv ${path}${cfg}/N.h5_0mom ${path}${cfg}/N.h5_0mom_1th_2000
#     mv ${path}${cfg}/BWZ.h5_0mom ${path}${cfg}/BWZ.h5_0mom_1th_2000
#     # break
# done

for cfg in `ls ${path}`
do 
    echo ${cfg}
    mv ${path}${cfg}/N.h5_0mom_2nd_1000 ${path}${cfg}/N.h5_0mom_2th_1000
    mv ${path}${cfg}/BWZ.h5_0mom_2nd_1000 ${path}${cfg}/BWZ.h5_0mom_2th_1000
    # break
done