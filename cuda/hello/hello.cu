#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cuda.h>

__global__ void vec_add(float* c, const float * a, const float * b, const int N){
  
  int tid = threadIdx.x + blockIdx.x*blockDim.x; //assume 1D thread grid
  if(tid<N){
  c[tid] = a[tid]+b[tid];
  }
}

int main(int argc, char* argv[]){
  // get command-line argument for vector size
  const int N = atoi(argv[1]);
  float * c = new float[N];
  float *a = new float[N];
  float * b = new float[N];
  
  //initialize a and b
  for(int i=0; i<N;i++){
    a[i] = i;
    b[i] = i;
  }

  // declare wonderful device arrays
  float * a_d;
  float * b_d;
  float * c_d;

  // allocate space for a,b, and c on the GPU
  cudaMalloc((void**) &a_d, N*sizeof(float));
  cudaMalloc((void**) &b_d, N*sizeof(float));
  cudaMalloc((void**) &c_d, N*sizeof(float));

  // transfer contents of a and b (on host) to a_d and b_d (on GPU)
  cudaMemcpy(a_d,a,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(b_d,b,N*sizeof(float),cudaMemcpyHostToDevice);

  // invoke calculation on the GPU
  const int BLOCK_SIZE=128;
  dim3 BLOCKS(BLOCK_SIZE,1,1);
  dim3 GRIDS((N+BLOCK_SIZE-1)/BLOCK_SIZE,1,1);
  vec_add<<<GRIDS,BLOCKS>>>(c_d,a_d,b_d,N);

  // return the result
  cudaMemcpy(c,c_d,N*sizeof(float),cudaMemcpyDeviceToHost);

  // verify that the result it is correct
  float sum = 0;
  for(int i = 0; i<N; i++){
     sum += c[i]; 
  }
  // should be equal to N*(N-1)
  std::cout << "Sum should be equal to "  << N*(N-1) << std::endl;
  std::cout << "Sum of vector result = "  << sum << std::endl;
  //COPY/PASTING THE ABOVE SECTIONS WILL CAUSE INCONSISTENCIES
  //(at least in vim) THAT YOU WILL HAVE TO MANUALLY EDIT (quotation marks for example) 

  //free memory on GPU
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  // be a good leader and free your memory!
  delete [] a;
  delete [] b;
  delete [] c;
  return 0;
}

