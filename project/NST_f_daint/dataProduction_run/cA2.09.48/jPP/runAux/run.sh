#!/bin/bash -l
#SBATCH --mail-user=li.yan@ucy.ac.cy
#SBATCH --job-name=<|cfg|>_jPP
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH -p normal
#SBATCH -C gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --account=s1174
#SBATCH --partition=normal
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
GPUPERNODE=4
NNODES=4


module load daint-gpu
module unload PrgEnv-cray
module load PrgEnv-gnu
module load cudatoolkit
module load CMake
module load cray-hdf5-parallel
. $(pwd)/../../runAux/module.sh

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

# Number of physical CPU cores per GPU

umask 002

#export LD_LIBRARY_PATH=$PROJECT_cecy00/bacchio1/quda_build_new/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/scratch/snx3000/fpittler/quda_library/libquda.so:${LD_LIBRARY_PATH}:/usr/local/cuda-11.2/targets/x86_64-linux/lib/nvrtc-prev/:/usr/local/cuda-11.2/targets/x86_64-linux/lib

export CUDA_DEVICE_MAX_CONNECTIONS=1
export QUDA_ENABLE_TUNING=1
export QUDA_TUNE_VERSION_CHECK=0
export QUDA_REORDER_LOCATION=GPU
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_GDR=1
export MPICH_RDMA_ENABLED_CUDA=1
ulimit -c 0


DIAGPREFIX="Diagram"
DATA_DIR="data"
OUTVECTOR="${DATA_DIR}/vector"

L=48
T=96

KAPPA=0.137290
CSW=1.57551
MUL=0.0009

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


XGRID=2
YGRID=2
ZGRID=2
TGRID=2

XDIM=$((L/XGRID))
YDIM=$((L/YGRID))
ZDIM=$((L/ZGRID))
TDIM=$((T/TGRID))

#MG opt ions
MG_NU_PRE="0 0 1 0"
MG_NU_POST="0 4 1 2"
MG_SETUP_TOL="5e-7"
MG_SETUP_ITER_0="0 1"
MG_SETUP_ITER_1="1 1"
MG_OMEGA=0.85
MG_SETUP_TYPE='null'
MG_PRE_ORTH=false
MG_POST_ORTH=true
MG_VERBOSITY=silent

MG_LEVELS=3
#Add more here for >2 levels
MG_N_VEC_0="0 24 1 32"
MG_BLK_SZE_0="0 6 6 3 3"
MG_MU_FACTOR_0="2 1.0"

MG_COARSE_SOLVER_1='2 ca-gcr'
MG_COARSE_TOL_1='1 0.22 0.22'
MG_COARSE_MAXITER_1='2 50'


echo "GRID(X,Y,Z,T) = ${XGRID} , ${YGRID} , ${ZGRID} , ${TGRID}"
echo "DIM(X,Y,Z,T)  = ${XDIM} , ${YDIM} , ${ZDIM} , ${TDIM}"
echo " "

echo `date`

export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=threads

cfg=<|cfg|>
echo Doing cfg ${cfg}
cfg_num=${cfg:1:4}
cfg_rep=${cfg:0:1}
cfg_rep_num=$(( $(printf "%d\n" "'$cfg_rep") -97))
seed=$(( (cfg_rep_num+1)*1000000+10#$cfg_num ))

QUDA_BIN=$(pwd)/../../runAux/Yan_jPP
CNF=/scratch/snx3000/fpittler/confs/cA2${cfg_rep}.09.48/conf.${cfg_num}

RUN_COMMAND="srun -n $((NNODES*GPUPERNODE)) ${QUDA_BIN} \
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
--Q-mg-setup-inv 0 cg 1 cg \
--Q-mg-mu-factor ${MG_MU_FACTOR_0} \
--Q-mg-omega ${MG_OMEGA} \
--Q-mg-setup-iters ${MG_SETUP_ITER_0} \
--Q-mg-pre-orth ${MG_PRE_ORTH} \
--Q-mg-post-orth ${MG_POST_ORTH} \
--Q-mg-verbosity 0 ${MG_VERBOSITY} 1 ${MG_VERBOSITY} \
--Q-mg-coarse-solver ${MG_COARSE_SOLVER_1} \
--Q-mg-coarse-solver-tol ${MG_COARSE_TOL_1} \
--Q-mg-coarse-solver-maxiter ${MG_COARSE_MAXITER_1} \
--Q-mg-smoother 0 ca-gcr 1 ca-gcr \
--Q-mg-smoother-tol 0 0.22 1 0.46 \
--Q-mg-nvec 0 24 1 32 \
--Q-mg-eig-nKr 2 2400 \
--Q-mg-eig-nEv 2 2048 \
--Q-mg-eig-nConv 2 2048 \
--Q-mg-eig 2 true \
--Q-mg-eig-type 2 trlm \
--Q-mg-eig-poly-deg 2 100 \
--Q-mg-eig-amin 2 4e-2 \
--Q-mg-eig-amax 2 8.0 \
--Q-mg-eig-max-restarts 2 25 \
--Q-mg-eig-use-dagger 2 false \
--Q-mg-eig-use-normop 2 true \
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