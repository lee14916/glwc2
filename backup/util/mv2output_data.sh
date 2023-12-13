#!/bin/bash

namePre=Diagram
namePost=.h5

for cfg in `cat runAux/conflist`; do
    echo ${cfg}
    mkdir -p output_data/${cfg}
    for file in run/${cfg}/${namePre}*${namePost}; do
        echo ${file}
        mv ${file} output_data/${cfg}/
    done
done

echo Done!