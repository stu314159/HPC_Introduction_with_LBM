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
  
  
  // initialize the dependent variable
  array f = exp(-(x*x));
  f((x< -5) & (x > -7))=1.; //it's sick that this works.
  array f_tmp = f;
  
  // create arrays of indices
  array ind = range(dim4(N));
  
  // x_p
  array x_p = constant(0,N,u32);
  x_p(seq(0,end-1,1)) = ind(seq(1,end,1));
  x_p((N-1))=0;
  
  // x_m
  array x_m = constant(0,N,u32);
  x_m(seq(1,end,1)) = ind(seq(0,end-1,1));
  x_m(0) = N-1;
  double C = u*dt/(2.*dx);
  
  af::Window window(512,512,"Moving Wave - Lax");
  
  auto t1 = high_resolution_clock::now(); // start the timer
  unsigned counter = 0;
  do{ 
    ++counter; // keep track of how many iterations I take.
    
    // Lax update
    f_tmp = 0.5*(f(x_p) + f(x_m)) - C*(f(x_p) - f(x_m));
    f = f_tmp; 
    
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
  std::cout << "Goodbye!" << std::endl;

  return 0;
}
