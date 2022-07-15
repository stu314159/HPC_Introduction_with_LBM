#ifndef CUDAPART_H
#define CUDAPART_H

#include <mpi.h>

class CudaPart
{
	public:
	CudaPart(int numEle,int rank, int size, MPI_Comm comm);
	~CudaPart();
	void print_result();//prints just the first three and last 3 numbers
     	void increment_on_gpu();

	private:
	
	void initialize();
	int numEle;
	int rank;
	int size;
	MPI_Comm comm;
	float * data = NULL;
	float * d_data = NULL;
};



#endif
