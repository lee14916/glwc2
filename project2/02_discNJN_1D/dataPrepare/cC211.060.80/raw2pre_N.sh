path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cC211.060.80/data_pre_N/

path=/p/arch1/hch02/hch02b/cC211.60.80/
for cfg in `ls ${path}`
do
    if [ ${#cfg} != 7 ] || ! [ ${cfg:4:2} == '_r' ]
    then
        continue
    fi
    echo ${cfg}
    mkdir -p ${path_pre}${cfg}
    for file in `ls ${path}${cfg}/twop_nucl_srcs650.h5`
    do
        # echo ${file}
        ln -s ${file} ${path_pre}${cfg}/N.h5_twop_nucl_srcs650
    done
    # break
done