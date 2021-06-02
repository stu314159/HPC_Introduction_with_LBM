__global__ void rusonov_update_p2(double * f_out, const double * f,const double * f_tmp, const double omega, const double nu, const int N)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (tid < N)
  {
    // get tid for neighbors
    int x_p = tid + 1;
    int x_2p = tid + 2;
    int x_m = tid - 1;
    int x_2m = tid - 2;
    
    // handle edge cases
    if (x_p == N)
    { 
      x_p = 0;
      x_2p = 1;
    }
    if (x_2p == N)
    {
      x_2p = 0;
    }
    
    if (x_m == -1)
    {
      x_m = N-1;
      x_2m = N-2;
    }
    
    if (x_2m == -1)
    { 
      x_2m = N-1;
    }
  
    f_out[tid] = f[tid]-(nu/24.0)*(-2.0*f[x_2p] + 7.0*f[x_p] - 7.0*f[x_m] + 2.0*f[x_2m]) 
             - (3.0*nu/8.0)*(f_tmp[x_p] - f_tmp[x_m]) 
             - (omega/24.0)*(f[x_2p] - 4.0*f[x_p] + 6.0*f[tid] - 4.0*f[x_m]+f[x_2m]);

  }
}
