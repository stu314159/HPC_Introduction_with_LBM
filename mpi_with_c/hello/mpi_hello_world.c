#include <mpi.h>
#include <cstdio>

int main(int argc, char** argv){
	// initialize the MPI environment
	MPI_Init(&argc,&argv);

	// get the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	// get the rank of the current process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	printf("Hello World from process %d out of %d.\n",rank,size);

	// clean up the MPI environment
	MPI_Finalize();

}
