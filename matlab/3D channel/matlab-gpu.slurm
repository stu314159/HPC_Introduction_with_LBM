#!/bin/sh

##SBATCH --mem-per-cpu=4gb
#SBATCH --time=02:00:00
#SBATCH -n 1                            # 1 instance of MATLAB
#SBATCH -p gpu                          # Run on GPU partition
#SBATCH --cpus-per-task=40              # 10 cores (1 node)
#SBATCH --gres=gpu:1                    # Request a single GPU

# Add MATLAB to system path
module load matlab/2020a

# Run code 
matlab -batch oblate_spheroid_cuda
