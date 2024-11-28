#!/bin/bash -l
#SBATCH --mail-user=li.yan@ucy.ac.cy
#SBATCH --job-name=<|cfg|>_jPP
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#SBATCH --time=02:00:00
#SBATCH --contiguous
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --account=s1174
#SBATCH --cpus-per-task=12
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err


echo "Report: Slurm Configuration"
echo "Job ID: ${SLURM_JOBID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Node cabinets and electric groups"
scontrol show nodes ${SLURM_JOB_NODELIST} | grep -i activefeatures | sort -u

HOME_DIR=$(pwd)
VERSION=0
NUM_THREADS=12
GPUPERNODE=1
NNODES=1

module load daint-gpu
module unload PrgEnv-cray
module load PrgEnv-gnu
module load cudatoolkit
module load CMake
module load cray-hdf5-parallel
. $(pwd)/../../runAux/module.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/project/s1174/fpittler/code/quda/build/lib
#source PizDaint_load_modules.sh

gdr=0
p2p=0
async=0
mempool=0

machine_id=PizDaint
quda_label=quda_1.0.x-dynamic_clover
quda_commit=e92ebd2e576691be6a5bea8a9b6e406b23f31246
gpu_arch=sm_60

export QUDA_RESOURCE_PATH=$(pwd)/../../runAux/${machine_id}-${quda_label}-${quda_commit}-${gpu_arch}_gdr${gdr}_p2p${p2p}
if [ ! -d ${QUDA_RESOURCE_PATH} ]; then
  mkdir -p ${QUDA_RESOURCE_PATH}
fi

export CRAY_CUDA_MPS=1
export OMP_NUM_THREADS=1
ulimit -c 0
echo " "

export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_DSLASH_COARSE_POLICY=0

export GOMP_CPU_AFFINITY=0-23:2
export QUDA_RESOURCE_PATH=${QUDA_RESOURCE_PATH}
export OMP_NUM_THREADS=1
export QUDA_ENABLE_GDR=${gdr}
export QUDA_ENABLE_P2P=${p2p}
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=${mempool}
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_NEMESIS_ASYNC_PROGRESS=${async}

DIAGPREFIX="Diagram"
DATA_DIR="data"
OUTVECTOR="${DATA_DIR}/vector"
LOADPROP="${DATA_DIR}/prop"

L=24
T=48

KAPPA=0.1400645
CSW=1.74
MUL=0.0053

INV_TOL=1.0e-9

QSQ_MAX=0
DSLASH_TYPE=twisted-clover
SOLVE_TYPE=direct-pc
MASS_NORM=mass
PIPELINE=24
NGCRKRYLOV=24
NITER=1000
VERIFY=false

PREC=double
PREC_SLOPPY=single
PREC_PRECON=half
RECON=12
RECON_SLOPPY=8
RECON_PRECON=8

UseFullOp=false
UseEven=true

# smearing parameters
alphaGauss=4.
nsmearAPE=50
nsmearGauss=50
alphaAPE=0.5
# File options
CORR_FILE_FORMAT=hdf5
CHECK_CORR_FILES=yes
CORR_WRITE_SPACE=momentum
# High mom format in yet implemented in the plugin
#HighMomForm=yes
VERBOSITY_LEVEL=verbose

# Determine number of nodes, gridsizes and dims

ZGRID=1
YGRID=1
TGRID=2
XGRID=1

XDIM=$((L/XGRID))
YDIM=$((L/YGRID))
ZDIM=$((L/ZGRID))
TDIM=$((T/TGRID))

#MG options

MG_NU_PRE="0 0 1 0"
MG_NU_POST="0 5 0 4"
MG_SETUP_TOL="5e-7"
MG_SETUP_ITER_0="0 1"
MG_SETUP_ITER_1="1 1"
MG_OMEGA=0.85
MG_SETUP_TYPE='null'
MG_PRE_ORTH=false
MG_POST_ORTH=true

MG_VERBOSITY=silent

MG_LEVELS=2
#Add more here for >2 levels
MG_N_VEC_0="0 24"
MG_BLK_SZE_0="0 4 4 4 4"
MG_MU_FACTOR_0="2 1.0"

MG_COARSE_SOLVER_1='1 gcr'
MG_COARSE_TOL_1='1 0.22'
MG_COARSE_MAXITER_1='1 10'

MG_SMOOTHER_TOL='0 0.25 1 0.25'


echo "GRID(X,Y,Z,T) = ${XGRID} , ${YGRID} , ${ZGRID} , ${TGRID}"
echo "DIM(X,Y,Z,T)  = ${XDIM} , ${YDIM} , ${ZDIM} , ${TDIM}"
echo " "

echo `date`

RND=RAND

cfg=<|cfg|>
echo Doing cfg ${cfg}
cfg_num=${cfg:1:4}
cfg_rep=${cfg:0:1}
cfg_rep_num=$(( $(printf "%d\n" "'$cfg_rep") -97))
seed=$(( (cfg_rep_num+1)*1000000+10#$cfg_num ))

QUDA_BIN=$(pwd)/../../runAux/Yan_jPP
# QUDA_BIN=/project/s1174/lyan/code/PLEGMA_main/build/plegma/Yan_jPP
CNF=/scratch/snx3000/fpittler/confs/cA211${cfg_rep}53.24/conf.${cfg_num}
# CNF=/project/s1174/lyan/code/scratch/run/tests/conf.${cfg_num}

RUN_COMMAND="srun --exclusive ${QUDA_BIN} \
  --procs ${XGRID} ${YGRID} ${ZGRID} ${TGRID} \
  --dims ${XDIM} ${YDIM} ${ZDIM} ${TDIM} \
  --verbosity 2  \
  --load-gauge ${CNF} \
  --flagFinish flagFinish \
  --seed_stoc ${seed} \
  --confnumber ${cfg_num} \
  --outdiagramPrefix ${DIAGPREFIX} \
  --momlistthreept-filename $(pwd)/../../runAux/combination_3pt \
  --nsmear-APE ${nsmearAPE} \
  --alpha-APE ${alphaAPE} \
  --nsmear-gauss ${nsmearGauss} \
  --alpha-gauss ${alphaGauss} \
  --Q-dslash-type ${DSLASH_TYPE} \
  --Q-prec ${PREC} \
  --Q-prec-sloppy ${PREC_SLOPPY} \
  --Q-recon ${RECON} \
  --Q-recon-sloppy ${RECON_SLOPPY} \
  --Q-recon-precondition ${RECON_PRECON} \
  --Q-kappa ${KAPPA} \
  --Q-mu ${MUL} \
  --Q-csw ${CSW} \
  --Q-mass-normalization ${MASS_NORM} \
  --Q-pipeline ${PIPELINE} \
  --Q-ngcrkrylov ${NGCRKRYLOV} \
  --Q-niter ${NITER} \
  --Q-tol ${INV_TOL} \
  --Q-mg-levels ${MG_LEVELS} \
  --Q-mg-block-size ${MG_BLK_SZE_0} \
  --Q-mg-nu-pre ${MG_NU_PRE} \
  --Q-mg-nu-post ${MG_NU_POST} \
  --Q-mg-setup-tol ${MG_SETUP_TOL} \
  --Q-mg-mu-factor ${MG_MU_FACTOR_0} \
  --Q-mg-omega ${MG_OMEGA} \
  --Q-mg-setup-iters ${MG_SETUP_ITER_0} \
  --Q-mg-pre-orth ${MG_PRE_ORTH} \
  --Q-mg-post-orth ${MG_POST_ORTH} \
  --Q-mg-setup-inv 0 cg \
  --Q-mg-verbosity 0 ${MG_VERBOSITY} 1 ${MG_VERBOSITY} \
  --Q-mg-coarse-solver ${MG_COARSE_SOLVER_1} \
  --Q-mg-coarse-solver-tol ${MG_COARSE_TOL_1} \
  --Q-mg-coarse-solver-maxiter ${MG_COARSE_MAXITER_1} \
  --Q-mg-smoother-tol ${MG_SMOOTHER_TOL} \
  --Q-mg-eig-nKr 1 384 \
  --Q-mg-eig-nEv 1 256 \
  --Q-mg-eig-nConv 1 256 \
  --Q-mg-eig 0 false 1 true \
  --Q-mg-preserve-deflation true \
"

#srun --tasks 4 bash -c 'echo "Rank: $PMI_RANK   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"' | sort

echo "Run command is:"
echo ${RUN_COMMAND}
echo " "
echo `date`
echo " "
eval ${RUN_COMMAND}
echo " "
echo `date`
echo " "