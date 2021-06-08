__global__ void bounceBackD2Q9(double * fOut, const double * fIn, const int * snl,
                               const int N_snl, const int N_lp)
{
  const int bb_spd[9] = {1, 4, 5, 2, 3, 8, 9, 6, 7};
  const int numSpd = 9;
  
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
