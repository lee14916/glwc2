#!/bin/bash

mkdir -p log

scriptName=postProduction-NJNpi

if [ $1 == "--division" ]; then
    timestamp=`date +%Y%m%d_%H%M%S`
    outputPre=log/${scriptName}_${timestamp}
    for index in $(eval echo {1..$2})
    do
        nohup python3 -u ${scriptName}.py --division $2 --index ${index} > ${outputPre}_id${index}.out &
    done
    return
fi