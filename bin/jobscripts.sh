#!/bin/bash
#SBATCH --account=icei-hbp-2021-0003  # icei-hbp-2022-0007
#SBATCH --nodes=1 # 1
#SBATCH --ntasks-per-node 120 # 120
#SBATCH --time=23:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.er
#SBATCH --mail-user=walid.sabouni@rwth-aachen.de
#SBATCH --mail-type=END
#SBATCH --partition=batch
#SBATCH --job-name=l2l-mc
#SBATCH --gres=gpu:0
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1


source /p/project/icei-hbp-2022-0007/Walid_2/activate.sh
python l2l-mc.py