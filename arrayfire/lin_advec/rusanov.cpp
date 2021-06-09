#include <arrayfire.h>
#include <iostream>
#include <cassert>

#include <chrono>


using namespace af;


int main(int argc, char** argv)
{

  af::info();
  
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  
  const unsigned N = 50000;
  const unsigned plot_freq = 2500;
  double u = 1.;
  
  const double x_left = -10;
  const double x_right = 10;
    
  // initialize the x-coordinate values  
  array x = seq(x_left,x_right,(x_right-x_left)/(N-1));
  
  
  double dx = (x_right - x_left)/(N-1);
  double dt = 0.6*dx/u;
  double nu = u*dt/dx;
  double omega = (4.*nu*nu+1.)*(4.-nu*nu)/5.;
  
  
  // initialize the dependent variable
  array f = exp(-(x*x));
  f((x< -5) & (x > -7))=1.; //it's sick that this works.
  array f_tmp = f;
  array f_nm = f;
  array f_tmp2 = f;
  
  // create arrays of indices
  array ind = range(dim4(N));
  
  // x_p
  array x_p = constant(0,N,u32);
  x_p(seq(0,end-1,1)) = ind(seq(1,end,1));
  x_p((N-1))=0;
  
  // x_2p
  array x_2p = constant(0,N,u32);
  x_2p(seq(0,end-2,1))=ind(seq(2,end,1));
  x_2p((N-2))=0;
  x_2p((N-1))=1;
  
  // x_m
  array x_m = constant(0,N,u32);
  x_m(seq(1,end,1)) = ind(seq(0,end-1,1));
  x_m(0) = N-1;
    
  // x_2m
  array x_2m = constant(0,N,u32);
  x_2m(seq(2,end,1))=ind(seq(0,end-2,1));
  x_2m(0)=N-2;
  x_2m(1)=N-1;
  
  af::Window window(512,512,"Moving Wave - Rusanov");
  
  auto t1 = high_resolution_clock::now(); // start the timer
  unsigned counter = 0;
  do{ 
    ++counter; // keep track of how many iterations I take.
    
    // Rusanov update
    f_nm = 0.5*(f(x_p) + f) - (nu/3.0)*(f(x_p)-f);
    f_nm.eval(); // ensure f_nm is evaluated before moving on
    f_tmp = f - (2.*nu/3.)*(f_nm - f_nm(x_m));
    f_tmp.eval(); // ensure f_tmp is evaluated before moving on
    f_tmp2 = f - (nu/24.)*(-2.*f(x_2p)+7.*f(x_p) - 7.*f(x_m) + 2.*f(x_2m))
             -(3.*nu/8.)*(f_tmp(x_p) - f_tmp(x_m))
             -(omega/24.)*(f(x_2p) - 4.*f(x_p)+6.*f(x_m) - 4.*f(x_m) +f(x_2m));
    f = f_tmp2;
    
    // if it's time to plot, call the plot function 
    if (counter % plot_freq == 0)
    {
      window.plot(x,f); 
      window.show();
    }   
    
  } while (!window.close()); // run until I'm ready to close the window
  
  auto t2 = high_resolution_clock::now();// stop the timer
  
  duration<double,std::milli> ms_double = t2 - t1;
  
  std::cout << "total time steps: " << counter << std::endl;
  double ex_time_per_dof = (ms_double.count()/1000.)/((double)(N*counter));
  std::cout << "Execution time per DOF: " << ex_time_per_dof << std::endl;
  std::cout << "Good by!" << std::endl;

  return 0;
}
