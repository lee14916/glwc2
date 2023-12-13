for cfg in `ls data_post`
do  
    if [ ! -f data_pre/${cfg}/N.h5_hch02k_twop ]
    then
        echo ${cfg}
        rm data_post/${cfg}/N.h5_hch02k_twop
    fi
done