__global__ void maccormack_update(double * f_out, const double * f, const double u, const double dx, const double dt, const int N)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < N) 
  {
    int x_p = tid + 1;
    int x_m = tid - 1;
    
    if (x_p >= N)
    {
      x_p = 0;
    }
    if (x_m < 0)
    {
      x_m = N-1;
    }
    
    double f_tmp1 = f[tid] - (u*dt/dx)*(f[x_p]-f[tid]);
    double f_tmp1_xm = f[x_m] - (u*dt/dx)*(f[tid]-f[x_m]); // f_tmp1 for x_m
    f_out[tid] = 0.5*(f[tid]+f_tmp1 - (u*dt/dx)*(f_tmp1 - f_tmp1_xm));
  
  }
}
