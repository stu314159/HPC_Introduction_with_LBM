#include <arrayfire.h>
#include <af/cuda.h>

#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

//CUDA includes
#include <cuda_runtime.h>

// My Includes
#include "lbm_lib.h"

using namespace af;
using namespace std;

int main(int argc, char** argv)
{
  af::info();
  
  // the user may forget an input; remind him/her
  if(argc<6){
    cout << "Fewer than 5 input arguments detected!" << endl;
    cout << "Usage: >>ldc2D [Re] [N] [TS] [omega] [dataFreq] where:" << endl;
    cout << "[Re] = flow Reynolds number." << endl;
    cout << "[N] = Number of lattice points along the cavity." << endl;
    cout << "[TS] = Number of time steps to perform." << endl;
    cout << "[omega] = relaxation parameter." << endl;
    cout << "[dataFreq] = # time steps between data outputs" << endl;
    cout << "Exiting the program.  Please try again." << endl;
    exit(1);
  } 
  
  float Re = (float)atof(argv[1]);
  uint N = (uint)atoi(argv[2]);
  uint numTs = (uint)atoi(argv[3]);
  float omega = (float)atof(argv[4]);
  uint dataFrequency = (uint)atoi(argv[5]);

  cout << "Re = " << Re << endl;
  cout << "N = " << N << endl;
  cout << "numTs = " << numTs << endl;
  cout << "omega = " << omega << endl;


  //for this problem, we know that there will be N*N lattice points
  // with 9 lattice directions per point.  
  // make use of this knowledge to simplify the code:
  const uint nnodes = N*N;
  const uint numSpd = 9;

   //basic host-side data arrays
  float* fDD = new float[nnodes*numSpd];
  int* ndType = new int[nnodes];
  float* ux = new float[nnodes];
  float* uy = new float[nnodes];
  float* pressure = new float[nnodes];
  float* uMag = new float[nnodes];

  float* xCoord = new float[nnodes];//lattice coordinates
  float* yCoord = new float[nnodes];

  // to simplify your life, I also do not allow you to pick the fluid.
  const float rho = 965.3; // density
  const float nu = 0.06/rho; // kinematic viscosity
  
  // get coordinate information; populate xCoord and yCoord
  LDC2D_getGeometry(xCoord,yCoord,1.0,1.0,N);

  // host side scaling and BC variables
  float u;
  float u_conv;
  float t_conv;
  float p_conv;
  float l_conv;

  // call setup function to initialize fDD and ndType
  // as well as to get scaling data
  LDC2D_setup(Re,N,omega,rho,nu,
		  fDD,ndType,u,
		  u_conv,t_conv,p_conv);
  l_conv = u_conv*t_conv;
  
  // declare AF arrays for calculations
  array af_fDD(nnodes*numSpd,fDD);
  array fEven = af_fDD;
  array fOdd = af_fDD;    
  array af_ndType(nnodes, ndType);  
  array af_ux(nnodes); //note this is uninitialized
  array af_uy(nnodes); // ditto 
  array af_pressure(nnodes);
  array af_w = constant(0,nnodes);
  array af_z = constant(0,nnodes);
 
  
  
  
  
  
  
  // be a good leader; free your memory
  delete [] fDD;
  delete [] ndType;
  delete [] ux;
  delete [] uy;
  delete [] pressure;
  delete [] uMag;
  delete [] xCoord;
  delete [] yCoord;

  std::cout << "Goodbye!" << std::endl;
  return 0;
}
