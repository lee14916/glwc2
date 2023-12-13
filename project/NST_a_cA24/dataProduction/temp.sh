for cfg in `ls data_post`
do  
    echo ${cfg}
    for file in `ls data_post/${cfg}/T*`
    do
        echo ${file}
        rm ${file}
    done
    # break
done


for cfg in `ls data_avgsrc`
do  
    echo ${cfg}
    for file in `ls data_avgsrc/${cfg}/T*`
    do
        echo ${file}
        rm ${file}
    done
    # break
done