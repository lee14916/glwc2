#!/bin/bash

if [ -z "$1" ]; then
    nohup python3 -u avgSrc.py > nohup.out &
    return
fi

mkdir -p log

if [ $1 == "--merge" ]; then
    timestamp=`date +%Y%m%d_%H%M%S`
    outputPre=log/avgSrc_${timestamp}
    # for patstodo in B W Z B_2pt W_2pt Z_2pt M_correct_2pt P_2pt,N_2pt,T,D1ii  
    for patstodo in N_2pt-pi0f-j D1ii-j,B N_2pt-j-pi0i,N_2pt-j"&"pi0i T-j,B_2pt W_2pt,Z_2pt M_correct_2pt,N_2pt-pi0f-pi0i,Z T-pi0f,D1ii-pi0i,N_2pt-j"&"sigmai N_2pt-j,D1ii,N_2pt-pi0f,T,N_2pt-pi0i,N_2pt,N_2pt-j-sigmai,W
    # for patstodo in N_2pt,N_2pt-pi0i,N_2pt-pi0f,T,D1ii,N_2pt-j
    do
    nohup python3 -u avgSrc.py --division -1 --index -1 --patstodo ${patstodo} > ${outputPre}_merge_${patstodo}.out &
    done
    return
fi

if [ $1 == "--division" ]; then
    timestamp=`date +%Y%m%d_%H%M%S`
    outputPre=log/avgSrc_${timestamp}
    for index in $(eval echo {1..$2})
    do
        nohup python3 -u avgSrc.py --division $2 --index ${index} > ${outputPre}_id${index}.out &
    done
    return
fi
