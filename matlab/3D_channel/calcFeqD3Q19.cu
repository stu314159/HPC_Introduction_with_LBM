__global__ void calcFeqD3Q19(double * fEq, const double * ux, const double * uy,
                            const double * uz, const double * rho, const int N)
{
  const double ex[19] = {0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0 -1.0,
                         1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
  const double ey[19] = {0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 
                        -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0};
                        
  const double ez[19] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
                         
                         
  const double w[19] = {3.0/9.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,
                        1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,
                        1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
  const int numSpd = 19;
  
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < N)
  {
    double cu;
    for(int spd=0;spd<numSpd;spd++)
    {
      cu = 3.0*(ex[spd]*ux[tid]+ey[spd]*uy[tid]+ez[spd]*uz[tid]);
      fEq[N*spd+tid] = w[spd]*rho[tid]*(1.0+cu+(0.5)*cu*cu - 
                       (1.5)*(ux[tid]*ux[tid] + uy[tid]*uy[tid] 
                       + uz[tid]*uz[tid]));
    }
  }
}
