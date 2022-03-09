module load GCC/9.3.0
# module load Intel/2020.2.254-GCC-9.3.0  ParaStationMPI/5.4.7-1 
module load OpenMPI/4.1.0rc1
module load mpi4py/3.0.3-Python-3.8.5 
module load SciPy-Stack/2020-Python-3.8.5 
module load CMake/3.18.0
module load GCCcore/.9.3.0
module load scikit/2020-Python-3.8.5
module load Java/15.0.1
module load PyTorch

export PYTHONPATH=$PYTHONPATH:<path_to_L2L>/:<path_to_sdict>/:<path_to_jube>/<path_to_nest_build>/install/lib64/python3.8/site-packages
