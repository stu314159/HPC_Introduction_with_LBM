#include "mex.h"
#include "gpu/mxGPUArray.h"

void __global__ lax(double * const f_tmp, double const * const f, const double C,
                    const int N)
{
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid < N)
  {
    int x_p = tid+1;
    int x_m = tid-1;
    
    if (x_p == N)
    {
      x_p = 0;
    }
    
    if (x_m < 0)
    {
      x_m = N-1;
    }
    
    f_tmp[tid] = 0.5*(f[x_p] + f[x_m]) - C*(f[x_p]-f[x_m]);
    
  }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  // I should have one argument on the lhs and three arguments on the right.
  //const mxGPUArray * const f; // input array
  //const double * const d_f;
  
  mxGPUArray * f_tmp; // output array
  double * d_f_tmp;
  
  //const double C; // u*dt/(2*dx) = C
  //const int N; // number of points
  
  const int TPB = 128;
  int Blocks;
  
  const mxGPUArray * const f = mxGPUCreateFromMxArray(prhs[0]);
  const double * const d_f = (const double * const)(mxGPUGetDataReadOnly(f));
  
  const double C = mxGetScalar(prhs[1]);
  const int N = mxGetScalar(prhs[2]);
  
  f_tmp = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(f),
                              mxGPUGetDimensions(f),
                              mxGPUGetClassID(f),
                              mxGPUGetComplexity(f),
                              MX_GPU_DO_NOT_INITIALIZE);
  d_f_tmp = (double *)(mxGPUGetData(f_tmp));
  
  Blocks = (N + TPB - 1)/TPB; //the usual trick...
  lax<<<Blocks,TPB>>>(d_f_tmp,d_f,C,N);
  
  
  plhs[0]=mxGPUCreateMxArrayOnGPU(f_tmp);
  
  mxGPUDestroyGPUArray(f);
  mxGPUDestroyGPUArray(f_tmp);

}
