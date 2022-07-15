#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include "CudaPart.h"

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	printf("Hello World from process %d out of %d.\n",rank,size);

	int N = atoi(argv[1]); // total length of array

	// what size of the array to I own
	int n = N/size; 
	if (rank < (N%size))
		n++; 

	CudaPart myPart = CudaPart(n,rank,size);

	printf("Process %d gets %d elements.\n",rank,n);

	myPart.increment_on_gpu();
	myPart.print_result();

	MPI_Finalize();

}
