__global__ void streamD2Q9(double * fIn, const double * fOut, const int * stm, const int N)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int numSpd = 9;
  
  if (tid < N)
  {
    for(int spd = 0; spd < numSpd; spd++)
    {
      int dof = N*spd + tid;
      int tgt_dof = N*spd + stm[dof];
      fIn[tgt_dof] = fOut[dof];
    }  
  }
}
