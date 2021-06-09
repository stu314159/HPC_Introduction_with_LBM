#include <arrayfire.h>
#include <iostream>
#include <cassert>


using namespace af;


int main(int argc, char** argv)
{

  af::info();
  
  
  const unsigned N = 25;
  const unsigned Num_ts = 10;
  double u = 1.;
  
  const double x_left = -10;
  const double x_right = 10;
    
  // initialize the x-coordinate values  
  //array x = constant(0,N,f32);
  //linspace(x,x_left,x_right,N);
  array x = seq(x_left,x_right,(x_right-x_left)/(N));
  //af_print(x);
  
  double dx = (x_right - x_left)/(N-1);
  double dt = 0.6*dx/u;
  
  
  // initialize the dependent variable
  array f = exp(-(x*x));
  f((x< -5) & (x > -7))=1.; //it's sick that this works.
  //af_print(f);
  array f_tmp = f;
  
  array ind = range(dim4(N));
  //af_print(ind);
  array x_p = constant(0,N,u32);
  x_p(seq(0,end-1,1)) = ind(seq(1,end,1));
  x_p((N-1))=0;
  
  //af_print(x_p);
  
  array x_m = constant(0,N,u32);
  x_m(seq(1,end,1)) = ind(seq(0,end-1,1));
  x_m(0) = N-1;
  //af_print(x_m);
  af::Window window(512,512,"Moving Wave - Lax");
  do{
  
    for(int ts = 0; ts < Num_ts; ts++)
    {
      f_tmp = 0.5*(f(x_p) + f(x_m)) - (u*dt/(2.*dx))*(f(x_p) - f(x_m));
      f = f_tmp;  
      window.plot(x,f);    
    }
  } while (!window.close());
 
  
  
  
  std::cout << "Good by!" << std::endl;

  return 0;
}
