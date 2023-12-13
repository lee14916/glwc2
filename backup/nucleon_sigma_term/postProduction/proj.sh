#!/bin/bash

mkdir -p log
timestamp=`date +%Y%m%d_%H%M%S`
outputPre=log/proj_${timestamp}
for patstodo in D1ii-j,B N_2pt-j-pi0i,N_2pt-j"&"pi0i,N_2pt-pi0f-j T-j,B_2pt W_2pt Z_2pt M_correct_2pt,N_2pt-pi0f-pi0i,Z T-pi0f,D1ii-pi0i,N_2pt-j"&"sigmai N_2pt-j,D1ii,N_2pt-pi0f,T,N_2pt-pi0i,N_2pt,N_2pt-j-sigmai,W
do
nohup python3 -u proj.py --patstodo ${patstodo} > ${outputPre}_${patstodo}.out &
done
