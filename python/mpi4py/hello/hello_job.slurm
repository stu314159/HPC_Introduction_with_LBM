#!/bin/bash


#SBATCH --time=00:10:00
#SBATCH -N 2
#SBATCH -n 10
#SBATCH -p hpc
#SBATCH --export=ALL
#SBATCH --job-name=mpi4py_test
##SBATCH --workdir=/home/sblair/projects/HPC_Introduction_with_LBM/python/mpi4py/hello

mpirun -np 20 ./hello_mpi.py 

