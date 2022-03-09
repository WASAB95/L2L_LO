module load GCC
module load ParaStationMPI/5.4.7-1
module load CMake
module load CUDA/11.0
module load SciPy-Stack/2020-Python-3.8.5
module load scikit/2021-Python-3.8.5
module load PyTorch
module load torchvision/0.8.2-Python-3.8.5

export PYTHONPATH=$PYTHONPATH:<path_to_jube>/JUBE/:<path_to_L2L>/:<path_to_sdict>/
export PATH=$PATH:<path_to_jube>/bin/
# more debug info for the terminal
export LMOD_SH_DBG_ON=1

source <path_to_nest_build_folder>/bin/nest_vars.sh
