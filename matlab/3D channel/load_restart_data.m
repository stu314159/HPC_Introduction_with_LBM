function [u,v,w,rho] = load_restart_data()

filename = 'restart.h5';
u = h5read(filename,'/velocity/x');
v = h5read(filename,'/velocity/y');
w = h5read(filename,'/velocity/z');
rho = h5read(filename,'/density/rho');

end