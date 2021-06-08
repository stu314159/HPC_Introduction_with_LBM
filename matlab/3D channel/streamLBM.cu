__global__ void streamLBM(double * fIn, const double * fOut, const int * stm, 
                           const int numSpd, const int N)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
 
  
  if (tid < N)
  {
    for(int spd = 0; spd < numSpd; spd++)
    {
      int dof = N*spd + tid;
      int tgt_dof = N*spd + stm[dof]-1;// correct for off-by-one issue for MATLAB/C
      fIn[tgt_dof] = fOut[dof];
    }  
  }
}
