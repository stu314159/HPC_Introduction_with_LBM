__global__ void bounceBackD3Q19(double * fOut, const double * fIn, const int * snl,
                               const int N_snl, const int N_lp)
{
  const int bb_spd[19] = {1, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8, 15, 14, 
                          13, 12, 19, 18, 17, 16}; // using MATLAB numbering
  const int numSpd = 19;
  
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (tid < N_snl)
  {
    int g_lp = snl[tid] - 1; // off-by-one adjust
    
    for (int spd=0; spd<numSpd; spd++)
    {
      fOut[N_lp*spd+g_lp] = fIn[N_lp*(bb_spd[spd]-1)+g_lp];    
    }  
  }
}
