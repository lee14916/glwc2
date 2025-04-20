path_pre=/p/project/ngff/li47/code/projectData2/01_Nsgm/cB211.072.64_base/data_pre_NJN/

echo -n Ncfgs= > log/loadNJN.out
ls ${path_pre} | wc -l >> log/loadNJN.out
ls ${path_pre}*/* | xargs -I @ -P 10 head -c1 @ >> log/loadNJN.out &