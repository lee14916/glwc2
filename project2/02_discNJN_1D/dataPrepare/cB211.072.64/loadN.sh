# case="N.h5_twop_threep_dt20_64srcs"
# case="N.h5_twop_threep_2"
case="N.h5_hch02k_twop"
path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cB211.072.64/data_pre_N/

echo -n Nfiles= > log/loadN.out
ls ${path_pre}*/${case} | wc -l >> log/loadN.out
ls ${path_pre}*/${case} | xargs -I @ -P 10 head -c1 @ >> log/loadN.out &