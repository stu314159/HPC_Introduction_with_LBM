% moving_wave_vectorized_rusanov.m

clear
%clc
close('all')

N = 5000;
u = 1;

plot_freq = 10;
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


for ts = 1:Num_ts
    
    if(mod(ts,100)==0)
        fprintf('Executing time step number %d.\n',ts);
    end
    
    f_nm = (0.5).*(f(x_p) + f) - (nu/3).*(f(x_p) - f);
    f_tmp = f - (2*nu/3).*(f_nm - f_nm(x_m));
    f = f - (nu/24).*(-2*f(x_2p) + 7*f(x_p) - 7*f(x_m) + 2*f(x_2m)) ...
        -(3*nu/8).*(f_tmp(x_p) - f_tmp(x_m)) ...
        -(omega/24).*(f(x_2p) - 4*f(x_p) + 6*f - 4*f(x_m) + f(x_2m));
    
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