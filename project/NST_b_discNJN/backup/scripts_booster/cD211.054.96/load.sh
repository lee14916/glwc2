case="N.h5_twop_threep_3"

echo -n Nfiles= > log/load.out
ls data_pre/*/${case} | wc -l >> log/load.out
ls data_pre/*/${case} | xargs -n 1 -I @ -P 1 head -n1 @ >> log/load.out &