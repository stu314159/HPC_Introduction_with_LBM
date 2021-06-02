__global__ void rusonov_update_p1(double * f_tmp, const double * f, const double omega, const double nu, const int N)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (tid < N)
  {
    // get tid for neighbors
    int x_p = tid + 1;    
    int x_m = tid - 1;
    
    
    // handle edge cases
    if (x_p == N)
    { 
      x_p = 0;      
    }
  
    
    if (x_m == -1)
    {
      x_m = N-1;     
    }
      
  
    double f_nm = 0.5*(f[x_p]+f[tid])-(nu/3.0)*(f[x_p]-f[tid]);
    double f_nm_xm = 0.5*(f[tid]+f[x_m])-(nu/3.0)*(f[tid]-f[x_m]);
    f_tmp[tid] = f[tid] - (2.0*nu/3.0)*(f_nm - f_nm_xm);

  }
}
