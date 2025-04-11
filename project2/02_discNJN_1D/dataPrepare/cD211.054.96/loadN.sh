case="N.h5_twop_threep_3"
path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cD211.054.96/data_pre_N/

echo -n Nfiles= > log/loadN_${case}.out
ls ${path_pre}*/${case} | wc -l >> log/loadN_${case}.out
ls ${path_pre}*/${case} | xargs -I @ -P 10 head -c1 @ >> log/loadN_${case}.out &