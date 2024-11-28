for line in `squeue | grep lyan | awk '{print $1 "..." $8}'`
do 
    echo ${line} ${#line}
    if [ ${#line} != 18 ]
    then
        echo ${line:0:8}
        scancel ${line:0:8}
    fi
done