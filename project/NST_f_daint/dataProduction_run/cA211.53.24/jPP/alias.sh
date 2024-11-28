scratchDir=.
buildDir=/project/s1174/lyan/code/PLEGMA_main/build
buildName=Yan_jPP

source runAux/module.sh

alias mj8="(cd ${buildDir} && make -j 8 ${buildName})"
alias sbat=". do_run.sh t"
alias sq="squeue | grep debug"
alias squ="squeue | grep $USER"
# 
alias run="(mj8 && { sbat; sleep 1; sq; })"