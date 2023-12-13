path=/p/scratch/pines/fpittler/run/nucleon_sigma_term/cA2.09.48/NJN/outputdata/
for cfg in `ls ${path}`
do
    echo ${cfg}
    mkdir -p data_pre/${cfg}
    for file in `ls ${path}${cfg}/*N.h5`
    do
        src=${file:91:17}
        ln -s ${file} data_pre/${cfg}/N.h5_${src}
    done
    # break
done

path2=/p/project/pines/li47/code/scratch/run/nucleon_sigma_term/cA2.09.48/QuarkLoops_pi0_insertion/data_post/
for cfg in `ls data_pre`
do
    echo ${cfg}
    for file in ${path2}${cfg}/*insertLoop.h5;
    do
        ln -s ${file} data_pre/${cfg}/jLoop.h5
    done
    # break
done

echo remove

for cfg in `ls data_pre`
do
    if ! [ -f "data_pre/${cfg}/jLoop.h5" ]; then
        echo ${cfg}
        rm -r data_pre/${cfg}
    fi
done