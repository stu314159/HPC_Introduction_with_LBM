__global__ void lax_update(double * f_out, const double * f, const double u, const double dx, const double dt, const int N)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x; // use 1D thread/block grid
  if (tid < N)
  {
    int x_p = tid+1;
    int x_m = tid-1;
    
    if (x_p >= N)
    {
      x_p = 0;
    }
    
    if (x_m < 0)
    {
      x_m = N-1;
    }
    
    f_out[tid] = 0.5*(f[x_p]+f[x_m])-(u*dt/(2.*dx))*(f[x_p]-f[x_m]);
  
  }

}
