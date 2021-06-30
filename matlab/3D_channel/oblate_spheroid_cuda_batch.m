%circ_obst3D_cuda2.m
% include a cuda kernel implementation for streaming as the profiler
% indicates that this is the most time-consuming operation

% computes fEq using a cuda kernel; this is the next most time-consuming
% operation

% does bounce-back with a CUDA kernel.

clear
clc
close('all')

make_gold_standard = 0;
% for now, gold standard =
% D3Q19 lattice
% LBGK dynamics
% Regularized BCs
% Re = 5
% Ndivs = 21
% Num_ts = 1000
% sphere obstacle
% fluid = glycol

validation_check = 0; % set to 1 if you want to compare this run of code against gold standard

profile_code = 0;

load_restart = 0;
save_restart = 1;

% Turbulence Model Parameter
Cs = 50;

lattice_selection = 2; 
% 1 = D3Q15 %<-- for this test, only D3Q15 available
% 2 = D3Q19
% 3 = D3Q27

BC_type = 1;
% Regularized

dynamics = 1; %<-- just use LBGK for now
% 1 = LBGK
% 2 = RBGK
% 3 = MRT

entropic = 0;
% 0 = no
% 1 = yes

grate_on = 0;%<-- for now, grate_on = 0 


Num_ts = 200000;
ts_rep_freq = 1000;
plot_freq = 10000;

Re = 5000;
dt = 5.0e-4;
Ny_divs = 65;

Lx_p = 1;
Ly_p = 1;
Lz_p = 4;

obst_type = 'obl_sph';
% 'none'
% 'sphere'
% 'cylinder'
% 'wmb'
% 'obl_sph'


fluid = 2;
% 1 = glycerin
% 2 = glycol
% 3 = water
% 4 = fake fluid for benchmarks

switch fluid
    case 1
        rho_p = 1260;
        nu_p = 1.49/rho_p;
        
    case 2
        rho_p = 965.3;
        nu_p = 0.06/rho_p;
        
    case 3
        rho_p = 1000;
        nu_p = 1e-3/rho_p;
        
    case 4
        rho_p = 1000;
        nu_p = 0.01;
        
end

switch obst_type
   
    case 'none'
        Lo = Ly_p;
        
    case 'sphere'
        x_c = Lx_p/2;
        y_c = Ly_p/2;
        z_c = Lz_p/2;
        r = Ly_p/10;
        Lo = 2*r;
        Ao = 2*r; % < -- reference area for Cd calculations.
        
    case 'cylinder'
        y_c = Ly_p/2;
        z_c = Lz_p/2;
        r = Ly_p/10;
        Lo = 2*r;
    
    case 'wmb'
        z_c = Lz_p/2;
        x_c = Lx_p/2;
        h = Ly_p/5;
        y_c = h/2;
        Lo = h;
        
    case 'obl_sph'
        z_c = Lz_p/3;
        x_c = Lx_p/2;
        y_c = Ly_p/2.5;
        a = Lx_p/4;
        c = a/6;
        Lo = 2*a;
        p = -(30/180)*pi; % angle of attack
    
end

Uo = nu_p*Re/Lo;
To = Lo/Uo;
Uavg = Uo;

Ld = 1; Td = 1; Ud = (To/Lo)*Uavg;
nu_d = 1/Re;

dx = 1/(Ny_divs-1);
u_lbm = (dt/dx)*Ud;
nu_lbm=(dt/(dx^2))*nu_d;
omega = get_BGK_Omega(nu_lbm);

u_conv_fact = (dt/dx)*(To/Lo);
t_conv_fact = (dt*To);
l_conv_fact = dx*Lo;
p_conv_fact = ((l_conv_fact/t_conv_fact)^2)*(1/3)/(l_conv_fact^3); % <--for EOS type methods...

rho_lbm = rho_p*(l_conv_fact^3);
%rho_out = rho_lbm;

% generate LBM lattice
xm = 0; xp = Lx_p;
ym = 0; yp = Ly_p;
zm = 0; zp = Lz_p;

Ny = ceil((Ny_divs-1)*(Ly_p/Lo))+1;
Nx = ceil((Ny_divs-1)*(Lx_p/Lo))+1;
Nz = ceil((Ny_divs-1)*(Lz_p/Lo))+1;
dims = [Nz,Ny,Nx];

x_space = linspace(xm,xp,Nx);
vis_x_plane = x_space(ceil(Nx/2));

[gcoord,~,faces]=Brick3Dr2(xm,xp,ym,yp,zm,zp,Nx,Ny,Nz);
[nnodes,~]=size(gcoord);

Xp = reshape(gcoord(:,1),[Nx Ny Nz]);
Yp = reshape(gcoord(:,2),[Nx Ny Nz]);
Zp = reshape(gcoord(:,3),[Nx Ny Nz]);

switch lattice_selection
    
    case 1
        [w,ex,ey,ez,bb_spd]=D3Q15_lattice_parameters();
        lattice = 'D3Q15';
    case 2
        [w,ex,ey,ez,bb_spd]=D3Q19_lattice_parameters();
        lattice = 'D3Q19';
    case 3
        [w,ex,ey,ez,bb_spd]=D3Q27_lattice_parameters();
        lattice = 'D3Q27';
        
end


stm = genStreamTgtVec3Dr2(Nx,Ny,Nz,ex,ey,ez);

numSpd = length(w);

LatticeSize = [Nx Ny Nz];
LatticeSpeeds = [ex; ey; ez];

eps_l = l_conv_fact;
x_pref = 0.5*Lx_p;
y_pref = 0.5*Ly_p;
z_pref = 0.96*Lz_p;
p_ref_LP=find((abs(gcoord(:,1)-x_pref)<=(eps_l/2)) & (abs(gcoord(:,2)-y_pref)<=(eps_l/2)) & ...
    (abs(gcoord(:,3)-z_pref)<=(eps_l/2)));
if(~isempty(p_ref_LP))
    p_ref_LP=p_ref_LP(1);
else
    error('No Reference Pressure Point!!');
end


% find inlet nodes and outlet nodes
inl = find((gcoord(:,3)==zm));
onl = find((gcoord(:,3)==zp));

% find top and bottom solid nodes
snl = find((gcoord(:,2)==ym) | (gcoord(:,2)==yp));

switch obst_type
   
    case 'none'
        
        % nothing to do here...
        
    case 'sphere'
        obst_list = find(sqrt((gcoord(:,1)-x_c).^2 +(gcoord(:,2)-y_c).^2 + (gcoord(:,3)-z_c).^2)<r);
        snl = [snl; obst_list];
    
    case 'cylinder'
        obst_list = find(sqrt((gcoord(:,2)-y_c).^2 + (gcoord(:,3)-z_c).^2)<r);
        snl = [snl; obst_list];
        
    case 'wmb'
        obst_list = find((gcoord(:,2) < h) & (gcoord(:,1) > (x_c-h/2)) & ...
            (gcoord(:,1) < (x_c + h/2)) & ...
            (gcoord(:,3) > (z_c - h/2)) & ...
            (gcoord(:,3) < (z_c + h/2)));
        
        snl = [snl;obst_list]; 
        snl = unique(snl);
        
    case 'obl_sph'

        obl_sph_fun = @(x,y,z) ((x ).^2 + (z ).^2)./(a^2) + ...
            ((y ).^2)./(c^2) - 1;
        % shift coordinates to oblate spheroid center prior to rotation
        gcoord_r = gcoord - [x_c y_c z_c];
        rot_mat = [1 0 0; 0 cos(p) sin(p); 0 -sin(p) cos(p)];
        gcoord_r = gcoord_r*rot_mat';
        
        obst_list = ...
            find(obl_sph_fun(gcoord_r(:,1),gcoord_r(:,2),gcoord_r(:,3)) <= 0);
        snl = [snl;obst_list];
        snl = unique(snl);
    
end

% eliminate any intersection between inl and onl and the solid node lists
inl = setxor(inl,intersect(inl,snl));
onl = setxor(onl,intersect(onl,snl));



% compute inl and onl velocity boundary conditions
%Umax = (3/2)*u_lbm;
Umax = u_lbm;
by = Ly_p/2;
uz_bc = Umax*(1-((gcoord(inl,2) - by)/by).^2);

%rho_out = rho_lbm*ones(length(onl),1);
rho_out = rho_lbm;

switch BC_type
    
    case 1
        Pi_1_flat = zeros(nnodes,9);
        e_i = [ex; ey; ez];
        
        indir_p = find(e_i(3,:)==-1);% density distributions of known densities
        indir_0 = find(e_i(3,:)==0); % density distributions parallel to inlet boundary
        indir_m = find(e_i(3,:)==1); % unknown density distribution directions
        
        outdir_p = find(e_i(3,:)==1);
        outdir_0 = find(e_i(3,:)==0);
        outdir_m = find(e_i(3,:)==-1);
        
        
        
        Q_mn = zeros(3,3,numSpd);
        for i = 1:numSpd
            Q_mn(:,:,i)=e_i(:,i)*e_i(:,i)';
        end
        
        Q_flat = zeros(numSpd,9);
        for i = 1:numSpd
            q_tmp=Q_mn(:,:,i); q_tmp = q_tmp(:); q_tmp=q_tmp';
            Q_flat(i,:)=q_tmp;
        end
end

switch dynamics
    
    case 1% BGK
        
    case 2 % RBGK
        
        Pi_1_flat = zeros(nnodes,9);
        e_i = [ex; ey; ez];
        
        indir_p = find(e_i(3,:)==-1);% density distributions of known densities
        indir_0 = find(e_i(3,:)==0); % density distributions parallel to inlet boundary
        indir_m = find(e_i(3,:)==1); % unknown density distribution directions
        
        outdir_p = find(e_i(3,:)==1);
        outdir_0 = find(e_i(3,:)==0);
        outdir_m = find(e_i(3,:)==-1);
        
        
        
        Q_mn = zeros(3,3,numSpd);
        for i = 1:numSpd
            Q_mn(:,:,i)=e_i(:,i)*e_i(:,i)';
        end
        
        Q_flat = zeros(numSpd,9);
        for i = 1:numSpd
            q_tmp=Q_mn(:,:,i); q_tmp = q_tmp(:); q_tmp=q_tmp';
            Q_flat(i,:)=q_tmp;
        end
        
    case 3 % MRT
       
        M = getMomentMatrix(lattice);
        S = getEwMatrixMRT(lattice,omega);
        omega_op = M\(S*M);
        
        
end

fEq = zeros(nnodes,numSpd);

if load_restart == 1
    [ux,uy,uz,rho] =  load_restart_data();
    for i = 1:numSpd
        cu = 3*(ex(i)*ux+ey(i)*uy+ez(i)*uz);
        fEq(:,i)=w(i)*rho.*(1+cu+(1/2)*(cu.*cu) - ...
            (3/2)*(ux.^2 + uy.^2 + uz.^2));
    end
    fIn = fEq;
    fOut = fIn;
else
    % initialize to zero.
    fIn=(rho_lbm*ones(nnodes,numSpd)).*(repmat(w,nnodes,1));
    fOut = fIn; % just for initialization.
    rho = sum(fIn,2);
    ux = (fIn*ex')./(rho*u_conv_fact);
    uy = (fIn*ey')./(rho*u_conv_fact);
    uz = (fIn*ez')./(rho*u_conv_fact);
    
end





vis_nodes = find(gcoord(:,1)==vis_x_plane);





fprintf('Number of Lattice-points = %d.\n',nnodes);
fprintf('Number of time-steps = %d. \n',Num_ts);


fprintf('LBM viscosity = %g. \n',nu_lbm);
fprintf('LBM relaxation parameter (omega) = %g. \n',omega);
fprintf('LBM flow Mach number = %g. \n',u_lbm);

%input_string = sprintf('Do you wish to continue? [Y/n] \n');

%run_dec = input(input_string,'s');

run_dec = 'Y';

if ((run_dec ~= 'n') && (run_dec ~= 'N'))
    
    fprintf('Ok! Cross your fingers!! \n');
    ts_num=0; % for naming the visualization data files

    % do some more pre-time-stepping set-up
    tic;
    
    % send data to the GPU
    fIn = gpuArray(fIn);
    fOut = gpuArray(fOut);
    %fEq = gpuArray(fEq);
    ex = gpuArray(ex); ey = gpuArray(ey); ez = gpuArray(ez);
    %rho = gpuArray(rho);
    SNL = zeros(nnodes,1);
    SNL(snl) = 1;
    SNL = gpuArray(int32(SNL));
    snl = gpuArray(int32(snl));
    INL = zeros(nnodes,1);
    INL(inl) = 1;
    INL = gpuArray(int32(INL));
    ONL = zeros(nnodes,1);
    ONL(onl) = 1;
    ONL = gpuArray(int32(ONL));
   
    
    
    
    if profile_code == 1
     profile on
    end
    
    for ts=1:Num_ts
       
        if(mod(ts,ts_rep_freq)==0)
            fprintf('Executing time step number %d.\n',ts);
        end
        
       if(mod(ts,2)==1)
           D3Q19_RegBC_LBGK_turb(fOut,fIn,SNL,INL,u_lbm,ONL,rho_out,omega,Cs,Nx,Ny,Nz);
       else
           D3Q19_RegBC_LBGK_turb(fIn,fOut,SNL,INL,u_lbm,ONL,rho_out,omega,Cs,Nx,Ny,Nz);
       end
        
        if(mod(ts,plot_freq)==0)
            % plot something, plot something cool!!
            % compute macroscopic properties
            rho = sum(fIn,2);
            ux = (fIn*ex')./rho;
            uy = (fIn*ey')./rho;
            uz = (fIn*ez')./rho;
            ux(snl)=0; uy(snl)=0; uz(snl)=0;
            
            ux_h = ux./u_conv_fact;
            uy_h = uy./u_conv_fact;
            uz_h = uz./u_conv_fact;
            velmag = sqrt(ux_h.*ux_h + uy_h.*uy_h + uz_h.*uz_h);
            pressure_h = rho*p_conv_fact;
            p_offset = pressure_h(p_ref_LP);
            pressure_h = pressure_h - p_offset;
           
            pressure_h = gather(pressure_h);
            ux_h = gather(ux_h); uy_h = gather(uy_h); uz_h = gather(uz_h);
            velmag = gather(velmag);
            h5_filename=sprintf('out%d.h5',ts_num);
            xmf_filename=sprintf('data%d.xmf',ts_num);
            write_data_H5(h5_filename,nnodes,ux_h,uy_h,uz_h,velmag,pressure_h);
            writeXdmf_dp(dims,dx,xmf_filename,h5_filename);
            
%             save_velocityAndPressureVTK_binary(reshape(pressure_h,[Nx Ny Nz]),...
%                 ux_h,uy_h,uz_h,Xp,Yp,Zp,ts_fileName);
            
            ts_num=ts_num+1;
            
            
        end
        
    end
    
    ex_time = toc;
    fprintf('LPU/sec = %g.\n',nnodes*Num_ts/ex_time);
    
    if make_gold_standard == 1
        save('gold_standard.mat','fIn');
    end
    
    if save_restart == 1
        % ensure I have up-to-date data (in LBM units)
        rho = sum(fIn,2);
        ux = (fIn*ex')./rho;
        uy = (fIn*ey')./rho;
        uz = (fIn*ez')./rho;
        ux(snl)=0; uy(snl)=0; uz(snl)=0;
        
        ux_h = gather(ux); uy_h = gather(uy); uz_h = gather(uz);
        rho_h = gather(rho);
        
        write_restart_data(ux_h,uy_h,uz_h,rho_h,nnodes);
    end
    
    clear fOut fEq stm rho ux uy uz
    
    if validation_check == 1
        fprintf('Validation check, error = %g \n',validate(fIn));
    end
    
    if profile_code == 1
        profile viewer
    end
    

    
else
    fprintf('Run aborted.  Better luck next time!\n');
end


