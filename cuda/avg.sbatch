#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1  ## To request specific GPU (v100 or rtx8000)
#SBATCH --job-name=single
#SBATCH --mail-type=END
#SBATCH --time=2:00:00
#SBATCH --mem=10GB
#SBATCH --mail-user=weijian.feng@nyu.edu
#SBATCH --output=single%j.out
module purge

module load openmpi/intel/4.1.1
./cuda_avg