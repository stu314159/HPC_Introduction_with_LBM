#include <arrayfire.h>
#include <af/cuda.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <chrono>
#include <cmath>

//CUDA includes
#include <cuda_runtime.h>

// My Includes
#include "lbm_lib.h"
#include "vtk_lib.h"

using namespace af;
using namespace std;

int main(int argc, char** argv)
{
  af::info();
  
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  
  // the user may forget an input; remind him/her
  if(argc<7){
    cout << "Fewer than 5 input arguments detected!" << endl;
    cout << "Usage: >>ldc2D [Re] [N] [TS] [omega] [dataFreq] where:" << endl;
    cout << "[Re] = flow Reynolds number." << endl;
    cout << "[N] = Number of lattice points along the cavity." << endl;
    cout << "[TS] = Number of time steps to perform." << endl;
    cout << "[omega] = relaxation parameter." << endl;
    cout << "[dataFreq] = # time steps between data outputs" << endl;
    cout << "[vtk_out] = [1 = vtk output | 0 = no vtk output]" << endl;
    cout << "Exiting the program.  Please try again." << endl;
    exit(1);
  } 
  
  float Re = (float)atof(argv[1]);
  uint N = (uint)atoi(argv[2]);
  uint numTs = (uint)atoi(argv[3]);
  float omega = (float)atof(argv[4]);
  uint dataFrequency = (uint)atoi(argv[5]);
  bool vtk_out = (bool)atoi(argv[6]);

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
  array af_umag(nnodes);
  
  // host pointers
  float * ux_h;
  float * uy_h;
  float * uz_h;  uz_h = af_w.host<float>(); //dumb, but okay.
  float * pressure_h;
  float * z_h = af_z.host<float>(); // again, dumb, but okay.
  
  // define some utility variables for visualization
  string densityFileStub("pressure");
  string velocityFileStub("velocityMagnitude");
  string vectorVelFileStub("velocity");
  string velocityCSVFileStub("velocityMagCSV");
  string fileSuffix(".vtk");
  string fileSuffixCSV(".csv");
  stringstream ts_ind;
  string ts_ind_str;
  int vtk_ts = 0;
  string fileName1;
  string fileName2;
  string fileName3;
  string fileNameCSV;
  string dataName1("pressure");
  string dataName2("velocityMagnitude");
  string dataName3("velocity");
  int dims[3];
  dims[0]=N; dims[1]=N; dims[2]=1;
  float origin[3];
  origin[0]=0.; origin[1]=0.; origin[2]=0.;
  float spacing[3];
  spacing[0]=l_conv; spacing[1]=l_conv; spacing[2]=l_conv;
   
  af::Window myWindow(N,N,"Lid Driven Cavity");
  myWindow.setColorMap(AF_COLORMAP_PLASMA);
   
  
  auto t1 = high_resolution_clock::now(); // start the timer
  for(uint ts = 0; ts<numTs; ts++)
  {
    if((ts+1)%1000 == 0)
    {
      cout << "Executing time step " << (ts+1) << endl;
    }
    
    if(ts%2==0)
    {
      // use the lbm_lib timestep function and associated kernels
      LDC2D_timestep(fOdd.device<float>(),fEven.device<float>(),
                     omega, af_ndType.device<int>(),u,N);
      fOdd.unlock(); fEven.unlock(); af_ndType.unlock();      
    
    } else {
    
      LDC2D_timestep(fEven.device<float>(),fOdd.device<float>(),
                     omega, af_ndType.device<int>(),u,N);    
      fOdd.unlock(); fEven.unlock(); af_ndType.unlock();
    }
    
    if((ts%dataFrequency)==0)
    {
      
      LDC2D_getVelocityAndDensity(af_ux.device<float>(),af_uy.device<float>(),
                                  af_pressure.device<float>(),u_conv,p_conv,
                                  fEven.device<float>(),N);
      af_ux.unlock(); af_uy.unlock(); af_pressure.unlock(); fEven.unlock();
     
      
      // figure out how to plot the data on the GPU
      af_umag = sqrt(af_ux*af_ux + af_uy*af_uy);  
      af_umag.eval();    
      array img_umag = moddims(af_umag,N,N,1,1);
      
      img_umag *= 100.; // this hack makes the low Re flows visible in image
            
      img_umag.eval();
      myWindow.image(img_umag);
      
      if (vtk_out)
      {
        // transfer data to host and plot with vtk shit
        ux_h = af_ux.host<float>(); uy_h = af_uy.host<float>(); 
        pressure_h = af_pressure.host<float>();
      
        // compute velocity magnitude (on the host)
        LDC2D_getVelocityMagnitude(uMag,ux_h,uy_h,N);
        // set pressure relative to the central lattice point (on the host)
        LDC2D_getRelativePressure(pressure_h,N);
      
        // set file names
	    ts_ind << vtk_ts; vtk_ts++;
	    fileName1 = densityFileStub+ts_ind.str()+fileSuffix;
	    fileName2 = velocityFileStub+ts_ind.str()+fileSuffix;
	    fileName3 = vectorVelFileStub+ts_ind.str()+fileSuffix;
	    fileNameCSV= velocityCSVFileStub+ts_ind.str()+fileSuffixCSV;
	    ts_ind.str("");

	    // output data file.
	    SaveVTKImageData_ascii(pressure_h,fileName1,dataName1,origin,spacing,dims);
	    SaveVTKImageData_ascii(uMag,fileName2,dataName2,origin,spacing,dims);
        SaveVTKStructuredGridVectorAndMagnitude_ascii(ux_h,uy_h,uz_h,
                                                      xCoord,yCoord,z_h,
                                                      fileName3,dataName3,dims);
      }		 
    
    }
  
  
  }
  auto t2 = high_resolution_clock::now();// stop the timer
  duration<double,std::milli> ms_double = t2 - t1;
  double LPUs = ((double)nnodes)*((double)numTs)/(ms_double.count()/1000.);
  
  cout << "Execution time: " << ms_double.count()/1000. << " seconds." << endl;
  cout << "Approximate Lattice Point Updates per Second: " << LPUs << endl;
  
  // be a good leader; free your memory
  
  if (vtk_out)
  {
    freeHost(pressure_h);
    freeHost(ux_h);
    freeHost(uy_h);
    freeHost(uz_h);
    freeHost(z_h);
  }
  
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
