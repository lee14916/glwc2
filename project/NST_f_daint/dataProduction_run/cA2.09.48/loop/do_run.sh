#!/bin/bash

# nohup ./do_run.sh t > log/do_run.out &

for cfg in `cat runAux/cfgs_run$1 `; do 
  if [ ${cfg} == a0240 ]; then
      continue
  fi
  echo ${cfg}
  mkdir -p run/${cfg}

  cp runAux/run.sh run/${cfg}/run_${cfg}.sh

  cd run/${cfg}/
  sed -i "s/<|cfg|>/$cfg/g" run_${cfg}.sh
  sbatch run_${cfg}.sh
  cd - > /dev/null
done

echo Done!