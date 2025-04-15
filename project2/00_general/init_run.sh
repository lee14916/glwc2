#!/bin/bash

# nohup ./init_run.sh > log/init_run.out &

init

for cfg in `cat runAux/cfgs_run$1 `; do 
  # if [ ${cfg} == a0240 ]; then
  #     continue
  # fi
  echo ${cfg}

  mkdir -p run/${cfg}
  cp runAux/run.sh run/${cfg}/run_${cfg}.sh

  cd run/${cfg}/
  sed -i "s/<|cfg|>/$cfg/g" run_${cfg}.sh
  sbatch run_${cfg}.sh
  cd - > /dev/null
done

echo Done!