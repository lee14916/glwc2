module purge
module load cray-mpich
# module load rca
module load daint-gpu
module load Boost
module unload PrgEnv-cray
module load PrgEnv-gnu
module load CMake
module load cray-hdf5-parallel
module load cudatoolkit
#export LD_LIBRARY_PATH='/usr/local/cuda-11.0/lib64:/opt/cray/pe/hdf5-parallel/1.12.0.4/GNU/8.2/lib:/opt/cray/pe/mpt/7.7.18/gni/mpich-gnu/8.2/lib:/opt/cray/pe/perftools/21.09.0/lib64:/opt/cray/rca/2.2.20-7.0.3.1_3.15__g8e3fb5b.ari/lib64:/opt/cray/alps/6.6.67-7.0.3.1_3.18__gb91cd181.ari/lib64:/opt/cray/xpmem/default/lib64:/opt/cray/dmapp/default/lib64:/opt/cray/pe/pmi/5.0.17/lib64:/opt/cray/ugni/default/lib64:/opt/cray/udreg/default/lib64:/opt/cray/pe/libsci/20.09.1/GNU/8.1/x86_64/lib:/apps/daint/UES/jenkins/7.0.UP03/21.09/daint-gpu/software/Boost/1.78.0-CrayGNU-21.09-python3/lib:/opt/cray/rca/2.2.20-7.0.3.1_3.15__g8e3fb5b.ari/lib64:/opt/cray/rca/2.2.20-7.0.3.1_3.15__g8e3fb5b.ari/lib64/librca.a'
#export LD_LIBRARY_PATH=$LD_LIBRARBY_PATH:/opt/cray/rca/2.2.20-7.0.3.1_3.15__g8e3fb5b.ari/lib64
#echo ${LD_LIBRARY_PATH}
