__global__ void velocityBC_D2Q9(double * fIn, const double * ux, const double * uy, const double * rho,
                                const double * ux_p, const double * uy_p, const int * lnl, 
                                const int N_lnl, const int N_lp)
{ 
  const double ex[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
  const double ey[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
  const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
  const int numSpd = 9;
  
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < N_lnl)
  { 
    int g_lp = lnl[tid] - 1; // fix off-by-one bug
    double dx = ux_p[tid] - ux[g_lp];
    double dy = uy_p[tid] - uy[g_lp];
    double cu;
    for(int spd = 1; spd<numSpd; spd++)
    {
      cu = 3.0*(ex[spd]*dx + ey[spd]*dy);
      fIn[N_lp*spd+g_lp] += w[spd]*rho[g_lp]*cu;
    }                    
  
  }
} 
