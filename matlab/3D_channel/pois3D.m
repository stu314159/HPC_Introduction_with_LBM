%pois3D.m

clear
clc
close('all');

location = 'home';
% 'home' or 'office'

lattice_selection = 1; 
% 1 = D3Q15 %<-- for this test, only D3Q15 available
% 2 = D3Q19
% 3 = D3Q27

dynamics = 1;
% 1 = LBGK %<--- for this test, only LBGK available
% 2 = TRT
% 3 = MRT %<---- No model for D3Q27

entropic = 0; 
% 0 = no %<--- for this test, only non-entropic available
% 1 = yes

save_vtk=1;
% whether or not to save vtk files for visualization
% 0 = no
% 1 = yes

% give this the name you want to use for your vtk files
sim_name = 'pois3D';

initialization = 0;
% 0 = initialize fIn to zero speed
% 1 = initialize fIn to Poiseuille profile <--not used for this problem

Num_ts = 20000;
ts_rep_freq = 1000;
plot_freq = 1000;




Re = 25;
dt = 4e-3;
Ny_divs = 21;

Lx_p = 1;
Ly_p =1;
Lz_p = 5;


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

Lo = Ly_p;
%Lo = 2*R_cyl;
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
p_conv_fact = ((l_conv_fact/t_conv_fact)^2)*(1/3); %<-- this should work within the fluid...


rho_lbm = rho_p;
rho_out = rho_lbm;

% generate LBM lattice
xm = 0; xp = Lx_p;
ym = 0; yp = Ly_p;
zm = 0; zp = Lz_p;

Ny = ceil((Ny_divs-1)*(Ly_p/Lo))+1;
Nx = ceil((Ny_divs-1)*(Lx_p/Lo))+1;
Nz = ceil((Ny_divs-1)*(Lz_p/Lo))+1;

[gcoord,~,faces]=Brick3Dr2(xm,xp,ym,yp,zm,zp,Nx,Ny,Nz);
[nnodes,~]=size(gcoord);

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

% find solid nodes
snl = find((gcoord(:,2)==ym)|(gcoord(:,2)==yp));


inl = find(gcoord(:,3)==zm);
onl = find(gcoord(:,3)==zp);

% exclude solid nodes from inl and onl
inl = setxor(inl,intersect(inl,snl));
onl = setxor(onl,intersect(onl,snl));


% compute inl and onl velocity boundary conditions
Umax = (3/2)*u_lbm;
by = Ly_p/2; %bz = Lz_p/2;
%ux_bc = Umax*(1 - ((gcoord(inl,2)-by)/by).^2).*(1-((gcoord(inl,3)-bz)/bz).^2);
uz_bc = Umax*(1-((gcoord(inl,2)-by)/by).^2);

% initialize to zero.
fIn=(rho_lbm*ones(nnodes,numSpd)).*(repmat(w,nnodes,1));
fOut = fIn; % just for initialization.
fEq = zeros(nnodes,numSpd);

rho = sum(fIn,2);
ux = (fIn*ex')./(rho*u_conv_fact);
uy = (fIn*ey')./(rho*u_conv_fact);
uz = (fIn*ez')./(rho*u_conv_fact);

Vmag = sqrt(ux.*ux + uy.*uy + uz.*uz);
pressure_p = rho*p_conv_fact;
p_offset = pressure_p(p_ref_LP);
pressure_p = pressure_p-p_offset;


fprintf('Number of Lattice-points = %d.\n',nnodes);
fprintf('Number of time-steps = %d. \n',Num_ts);


fprintf('LBM viscosity = %g. \n',nu_lbm);
fprintf('LBM relaxation parameter (omega) = %g. \n',omega);
fprintf('LBM flow Mach number = %g. \n',u_lbm);

input_string = sprintf('Do you wish to continue? [Y/n] \n');

run_dec = input(input_string,'s');

if ((run_dec ~= 'n') && (run_dec ~= 'N'))
    
    fprintf('Ok! Cross your fingers!! \n');

    
    
     if(save_vtk==1)
            % save the initial vtk data
            ts_num=0;
            
            
            
            vtk_suffix=sprintf('_Velocity%d.vtk',ts_num);
            ts_fileName=strcat(sim_name,vtk_suffix);
            %         save_velocityAndPressureVTK_binary(pressure_p,ux,uy,uz,gcoord(:,1),...
            %             gcoord(:,2),gcoord(:,3),ts_fileName);
            save_VectorAndMagnitudeVTK_binary(ux,uy,uz,...
                gcoord(:,1),gcoord(:,2),gcoord(:,3),ts_fileName,'Velocity');
            
            vtk_suffix=sprintf('_pressure%d.vtk',ts_num);
            ts_fileName=strcat(sim_name,vtk_suffix);
            origin = [xm ym zm];
            spacing = [l_conv_fact l_conv_fact l_conv_fact];
            save_scalarStructuredPoints3D_VTK_binary(ts_fileName,'scalPressure',...
                reshape(rho,[Nx Ny Nz]),origin, spacing);
            
        end
        
        % addpaths for jacket and jacketSDK
    
    

 
        % commence time stepping
        
       for ts = 1:Num_ts
        if(mod(ts,ts_rep_freq)==0)
            fprintf('Executing time step number %d.\n',ts);
        end
        
        % compute density
        rho = sum(fIn,2);
        
        % compute velocities
        ux = (fIn*ex')./rho;
        uy = (fIn*ey')./rho;
        uz = (fIn*ez')./rho;
        
        % set macroscopic and Microscopic Dirichlet-type boundary
        % conditions
        
        % macroscopic BCs
        ux(inl)=0;
        uy(inl)=0;
        uz(inl)=uz_bc;
        
        ux(onl)=0;
        uy(onl)=0;
        uz(onl)=uz_bc;

        % microscopic BCs
        fIn(inl,:)=velocityBC_3D(fIn(inl,:),w,ex,ey,ez,...
            0*uz_bc,0*uz_bc,uz_bc);
        fIn(onl,:)=velocityBC_3D(fIn(onl,:),w,ex,ey,ez,...
            0*uz_bc,0*uz_bc,uz_bc);
        
        % Collide
        switch dynamics
            
            case 1 % LBGK
                for i = 1:numSpd
                    cu = 3*(ex(i)*ux+ey(i)*uy+ez(i)*uz);
                    fEq(:,i)=w(i)*rho.*(1+cu+(1/2)*(cu.*cu) - ...
                        (3/2)*(ux.^2 + uy.^2+uz.^2 ));
                    fOut(:,i)=fIn(:,i)-omega*(fIn(:,i)-fEq(:,i));
                end
                
            case 2 % TRT
                
                % find even part and odd part of off-equilibrium
                % for all speeds, then relax those parts...
                
                % compute fEq
                for i = 1:numSpd
                    cu = 3*(ex(i)*ux+ey(i)*uy+ez(i)*uz);
                    fEq(:,i)=w(i)*rho.*(1+cu+(1/2)*(cu.*cu) - ...
                        (3/2)*(ux.^2 + uy.^2+uz.^2 ));
                end
                
                % compute fNEq
                fNEq = fEq - fIn;
                
                % compute odd and even part of fNEq
                for i= 1:numSpd
                    fEven(:,i) = (1/2)*(fNEq(:,i)+fNEq(:,bb_spd(i)));
                    fOdd(:,i) = (1/2)*(fNEq(:,i)-fNEq(:,bb_spd(i)));
                    
                end
                
                % relax on these even and odd parts
                fOut = fIn + omega*fEven + fOdd;
                % omega for odd part is equal to 1.
                
            case 3 % MRT
                % compute fEq
                for i = 1:numSpd
                    cu = 3*(ex(i)*ux+ey(i)*uy+ez(i)*uz);
                    fEq(:,i)=w(i)*rho.*(1+cu+(1/2)*(cu.*cu) - ...
                        (3/2)*(ux.^2 + uy.^2+uz.^2 ));
                end
                % collide
                fOut = fIn - (fIn - fEq)*omega_op;
                
                
        end
        
        % bounce-back
        for i = 1:numSpd
            fOut(snl,i)=fIn(snl,bb_spd(i));
        end
        
        
        % stream
        for i = 1:numSpd
            fIn(stm(:,i),i)=fOut(:,i);
        end
        
        if(mod(ts,plot_freq)==0)
            
            if(save_vtk==1)
                
                ts_num=ts_num+1;
                
           
                
                % write to file
                vtk_suffix=sprintf('_Velocity%d.vtk',ts_num);
                ts_fileName=strcat(sim_name,vtk_suffix);
                %         save_velocityAndPressureVTK_binary(pressure_p,ux,uy,uz,gcoord(:,1),...
                %             gcoord(:,2),gcoord(:,3),ts_fileName);
                save_VectorAndMagnitudeVTK_binary(ux,uy,uz,...
                    gcoord(:,1),gcoord(:,2),gcoord(:,3),ts_fileName,'Velocity');
                
                vtk_suffix=sprintf('_pressure%d.vtk',ts_num);
                ts_fileName=strcat(sim_name,vtk_suffix);
                origin = [xm ym zm];
                spacing = [l_conv_fact l_conv_fact l_conv_fact];
                save_scalarStructuredPoints3D_VTK_binary(ts_fileName,'scalPressure',...
                    reshape(rho,[Nx Ny Nz]),origin, spacing);
                
            end
            
        end
       
 end 
            
            
            
           
             
               
            
     
    
else
    fprintf('Run aborted.  Better luck next time!\n');
end