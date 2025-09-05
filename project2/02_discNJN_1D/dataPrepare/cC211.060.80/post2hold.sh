base=/p/project1/ngff/li47/code/projectData/02_discNJN_1D/cC211.060.80
out=${base}/data_post_hold

# in=${base}/data_post_N
# for cfg in `ls ${in}`
# do
#     mkdir -p ${out}/${cfg}
#     for file in `ls ${in}/${cfg}`
#     do
#         mv ${in}/${cfg}/${file} ${out}/${cfg}
#     done
# done

in=${base}/cyclone
for cfg in `ls ${in}`
do
    mkdir -p ${out}/${cfg}
    for file in `ls ${in}/${cfg}`
    do
        mv ${in}/${cfg}/${file} ${out}/${cfg}
    done
done