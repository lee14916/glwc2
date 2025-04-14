path_pre=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cD211.054.96/data_pre_jsc/

echo -n Ncfgs= > log/loadjsc.out
ls ${path_pre} | wc -l >> log/loadjsc.out
ls ${path_pre}*/* | xargs -I @ -P 10 head -c1 @ >> log/loadjsc.out &