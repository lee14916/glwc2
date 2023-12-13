# case="N.h5_twop_threep_dt20_64srcs"
# case="N.h5_twop_threep_2"
case="N.h5_hch02k_twop"

echo -n Nfiles= > log/load.out
ls data_pre/*/${case} | wc -l >> log/load.out
ls data_pre/*/${case} | xargs -n 1 -I @ -P 1 head -c1 @ >> log/load.out &