#!/bin/bash

inPath=/project/s1174/lyan/code/scratch/run/nucleon_sigma_term/cA211b.30.32/NJNpi_N0pi+/
namePre=Diagram
namePost=.h5

for cfg in `cat ${inPath}runAux/conflist`; do
    echo ${cfg}
    mkdir -p output_data/${cfg}
    for file in ${inPath}output_data/${cfg}/*; do
        echo ${file}
        mv ${file} output_data/${cfg}/
    done
done

echo Done!