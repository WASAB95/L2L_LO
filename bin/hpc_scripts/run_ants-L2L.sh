#!/bin/bash -x
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=10:00:00
#SBATCH --output=output_%j.out  
#SBATCH --error=error_%j.er     
#SBATCH --mail-user=
##SBATCH --mail-type=END
#SBATCH --partition=batch
#SBATCH --job-name=antcolony

source ./set_ants-L2L.sh
python l2l-neuroevolution_ant_colony.py
