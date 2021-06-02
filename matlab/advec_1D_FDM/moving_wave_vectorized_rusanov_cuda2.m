% moving_wave_vectorized_rusanov.m

clear
%clc
close('all')

N = 50000;
u = 1;

plot_freq = 2500;
plot_switch = 1;
x_left = -10;
x_right = 10;
x_space = linspace(x_left,x_right,N);

dx = x_space(2)-x_space(1);
dt = 0.6*(dx)/u;
nu = u*dt/dx;
omega = (4*nu*nu+1)*(4-nu*nu)/5;
%Num_ts = min(1000,ceil(15/dt));
Num_ts = 5000;
% set initial condition
%f = zeros(N,1);
x_space = linspace(x_left,x_right,N);

f_l = 1;
f = f_l*exp(-(x_space.*x_space));
f((x_space < -5) & (x_space > -7)) = 1;


f_tmp = zeros(N,1);
f_nm = zeros(N,1);



% plot initial condition
plot(x_space,f,'-b');
axis([x_left x_right 0 1.1*f_l]);
grid on
drawnow

%title('\bf{Initial Condition}');


tic;

ind = (1:N)';
x_m = circshift(ind,1);
x_p = circshift(ind,-1);
x_2m = circshift(ind,2);
x_2p = circshift(ind,-2);


x_m = gpuArray(x_m);
x_p = gpuArray(x_p);
x_2m = gpuArray(x_2m);
x_2p = gpuArray(x_2p);

f_nm = gpuArray(f_nm);
f_tmp = gpuArray(f_tmp);
f = gpuArray(f');
f_out = zeros(N,1);
f_out = gpuArray(f_out);

% construct my cuda kernel
k1 = parallel.gpu.CUDAKernel('rusanov_update_p1.ptx','rusanov_update_p1.cu');
k2 = parallel.gpu.CUDAKernel('rusanov_update_p2.ptx','rusanov_update_p2.cu');

TPB = 32*4;
k1.ThreadBlockSize = [TPB,1,1];
k1.GridSize = [ceil(N/TPB),1,1];
k2.ThreadBlockSize = [TPB,1,1];
k2.GridSize = [ceil(N/TPB),1,1];

N = int32(N);

for ts = 1:Num_ts
    
    if(mod(ts,100)==0)
        fprintf('Executing time step number %d.\n',ts);
    end
    
   
    f_tmp = feval(k1,f_tmp,f,omega,nu,N);    
    f_out = feval(k2,f_out,f,f_tmp,omega,nu,N);
    f = f_out;
     
    
    if(plot_switch==1)
        if(mod(ts,plot_freq)==0)
            plot(x_space,f,'-b')
            axis([x_left x_right 0 1.1*f_l]);
            grid on
            title('\bf{Rusanov Method}','FontSize',12);
            drawnow
            
        end
    end
    
end

ex_time = toc;



plot(x_space,f,'-b');
axis([x_left x_right 0 1.1*f_l]);
title('\bf{Final Condition}');
grid on
drawnow

fprintf('Execution time = %g.\n Average time per DOF*update = %g. \n',ex_time, ex_time/(N*Num_ts));