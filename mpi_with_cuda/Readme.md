# MPI and CUDA
This example requires that you have access to a workstation with at least 1 GPU. It also requires (I am sorry to say) that for multi-GPU systems that the GPUs reside on different nodes; the code doesn't try to de-conflict between multiple devices on a single system.

Provided you have access to such a system, you also need your environment set so you have both NVIDIA and MPI compiler tools available.  Consult your system administrator if you need help with this.

## Steps to build this example:
Assuming you have already cloned the repository, navigate to the mpi_with_cuda folder and carry out the following steps:

1. mkdir build
2. cd build
3. cmake ..

This will result in the CMake build system examining your system and (hopefully!) create a build script in the build folder.  Assuming this has been accomplished:

4. make

This will build the executable.

To run, you would invoke something like:

mpiexec -n 4 ./hello 5000

This will run the code on 4 mpi processes (each with its own GPU) on an array of total length 5000.  
