for cfg in `ls data_post`
do
    echo ${cfg}
    for file in data_post/${cfg}/N.h5_twop_129srcs
    do
        # head -c1 ${file}
        echo ${file}
        mkdir -p data_post_old/${cfg}/
        mv ${file} data_post_old/${cfg}/
    done
done