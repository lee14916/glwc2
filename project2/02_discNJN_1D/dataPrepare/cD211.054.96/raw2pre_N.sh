path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cD211.054.96/data_pre_N/

path=/p/arch1/hch02/hch02b/cD211.54.96/twop_threep_3/
for file in ${path}twop*.h5
do
    cfg=${file:53:7}
    echo ${cfg}
    mkdir -p ${path_pre}${cfg}
    ln -s ${file} ${path_pre}${cfg}/N.h5_twop_threep_3
    # break
done