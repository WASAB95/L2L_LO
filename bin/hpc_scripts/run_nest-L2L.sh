#!/bin/bash -x
#SBATCH --account=
#SBATCH --nodes=7
#SBATCH --ntasks-per-node 16
#SBATCH --time=11:00:00
#SBATCH --output=output_%j.out  
#SBATCH --error=error_%j.er     
#SBATCH --mail-user=
#SBATCH --mail-type=END
# SBATCH --partition=batch
#SBATCH --job-name=nest_reservoir
#SBATCH --gres=gpu:0

source ./set_nest-L2L.sh
python ../l2l-snn-adaptive.py
