#include <arrayfire.h>
#include <af/cuda.h>

#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

using namespace af;

int main(int argc, char** argv)
{
  af::info();
  
  const unsigned ts_rep_freq = 500;
  const unsigned plot_freq = 1000;
  
  // geometry parameters
  const double Lx_p = 1;
  const double Ly_p = 1;
  
  // fluid parameters 
  const double rho_p = 965.3;
  const double nu_p = 0.06/rho_p;
  
  // flow parameters and non-dimensionalization
  const double Re = 1000;
  const double Uavg = nu_p*Re/Ly_p;
  const double Lo = Ly_p;
  const double Uo = Uavg;
  const double To = Lo/Uo;
  
  const double Ld = 1; const double Td = 1; const double Ud = (To/Lo)*Uavg;
  const double nu_d = 1.0/Re;
  
  // conversion to LBM units
  const double dt = 0.0002;
  const unsigned Ny_divs = 301;
  const double dx = 1.0/(Ny_divs - 1.0);
  const double u_lbm = (dt/dx)*Ud;
  const double nu_lbm = (dt/(dx*dx))*nu_d;
  const double omega = 1.0/(3*nu_lbm + 0.5);
  
  const double u_conv_fact = (dt/dx)*(To/Lo);
  const double t_conv_fact = (dt*To);
  const double rho_lbm = rho_p;
  const double rho_out = rho_lbm;
  
  // initialize the lattice
  const unsigned numSpd = 9;
  const unsigned Ny = Ny_divs;
  const unsigned Nx = ceil((Ny_divs-1)*(Lx_p/Ly_p))+1.0;
  const unsigned nnodes = Nx*Ny;

  std::cout << "Goodbye!" << std::endl;
  return 0;
}
