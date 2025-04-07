base=/p/project/ngff/li47/code/projectData/02_discNJN_1D/cB211.072.64
out=${base}/data_post_hold

in=${base}/data_post
for cfg in `ls ${in}`
do
    mkdir -p ${out}/${cfg}
    for file in `ls ${in}/${cfg}`
    do
        mv ${in}/${cfg}/${file} ${out}/${cfg}
    done
done

in=${base}/loop_cyclone
for cfg in `ls ${in}`
do
    mkdir -p ${out}/${cfg}
    for file in `ls ${in}/${cfg}`
    do
        mv ${in}/${cfg}/${file} ${out}/${cfg}
    done
done