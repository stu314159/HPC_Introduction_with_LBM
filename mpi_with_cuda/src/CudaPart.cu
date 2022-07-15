#include "CudaPart.h"
#include <cuda.h>
#include <cstdio>

__global__ void increment(float * d, const int N);


__global__ void increment(float * d, const int N)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if(tid < N)
  {
    d[tid]+=tid;
  }
}


CudaPart::CudaPart(int numEle, int rank, int size):
	numEle(numEle), rank(rank), size(size)
{
  initialize();
}

CudaPart::~CudaPart()
{
  // de-allocate memory on the host and device
  delete [] data;
  cudaFree(d_data);
}


void CudaPart::initialize()
{
  // allocate memory on the host
  data = new float[numEle];

  // initialize each entry the data to rank*1000
  for (int i = 0; i<numEle; i++)
  {
    data[i] = rank*1000.;
  }
  // allocate memory on the GPU device
  cudaMalloc((void**)&d_data,numEle*sizeof(float));

  // transfer data from the host to the device
  cudaMemcpy(d_data,data,numEle*sizeof(float),cudaMemcpyHostToDevice);

}

void CudaPart::increment_on_gpu()
{

  // create a 1D thread blcok grid with specified block size
  const int BLOCK_SIZE = 128;
  dim3 BLOCKS(BLOCK_SIZE,1,1);
  dim3 GRIDS((numEle+BLOCK_SIZE-1)/BLOCK_SIZE,1,1);

  increment<<<GRIDS,BLOCKS>>>(d_data,numEle);

  // copy data back to the CPU memory
  cudaMemcpy(data,d_data,numEle*sizeof(float),cudaMemcpyDeviceToHost);

}

void CudaPart::print_result()
{
  printf("Rank %d: %f, %f, %f,...,%f,%f,%f\n",rank,
		  data[0],data[1],data[2],
		  data[numEle-3],data[numEle-2],data[numEle-1]);
}
