buildDir=/capstor/store/cscs/userlab/s1174/lyan/code/PLEGMA/build/
buildName=Yan_NpiScattering_Wilson

init
source runAux/module.sh

alias mj8="(cd ${buildDir} && make -j 8 ${buildName})"
alias sq="squeue -p debug"
alias squ="squeue | grep $USER"