% moving_wave_vectorized_lax.m
%
%  Use custome *.cu file for the update step and compile into *.ptx
%  then call within the code
%

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
nu = u*dx/dt;

%Num_ts = min(1000,ceil(15/dt));
Num_ts=5000;
% set initial condition
%f = zeros(N,1);
x_space = linspace(x_left,x_right,N);

f_l = 1;
f = f_l*exp(-(x_space.*x_space));
f((x_space < -5) & (x_space > -7)) = 1;

f_tmp = zeros(N,1);

% plot initial condition
plot(x_space,f,'-b');
axis([x_left x_right 0 1.1*f_l]);
title('\bf{Initial Condition}');
drawnow


tic;

ind = (1:N)';
x_m = circshift(ind,1);
x_p = circshift(ind,-1);

x_m = gpuArray(x_m);
x_p = gpuArray(x_p);
f_tmp = gpuArray(f_tmp);
f = gpuArray(f);

% construct my cuda kernel
k = parallel.gpu.CUDAKernel('lax_update.ptx','lax_update.cu');
% k takes arguments: f_out, f, u, dx, dt, N
TPB = 32*4;
k.ThreadBlockSize = [TPB,1,1];
k.GridSize = [ceil(N/TPB),1,1];
N = int32(N);


%profile on
for ts = 1:Num_ts
    
    if(mod(ts,100)==0)
       fprintf('Executing time step number %d.\n',ts);
    end
    
    %f_tmp = 0.5.*(f(x_p)+f(x_m))-(u*dt/(2*dx)).*(f(x_p)-f(x_m));
    f_tmp = feval(k,f_tmp,f,u,dx,dt,N);
    f = f_tmp;
    
    if(plot_switch==1)
        if(mod(ts,plot_freq)==0)
            plot(x_space,f,'-b')
            axis([x_left x_right 0 1.1*f_l]);
            title('\bf{Lax Method}','FontSize',12);
            grid on
            drawnow
        end
    end
    
end

ex_time = toc;
%profile viewer
plot(x_space,f,'-b');
axis([x_left x_right 0 1.1*f_l]);
title('\bf{Final Condition}');
grid on
drawnow

fprintf('Execution time = %g.\n Average time per DOF*update = %g. \n',ex_time, ex_time/(N*Num_ts));