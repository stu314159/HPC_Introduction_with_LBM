#python code for 2D LDC implementation

# python package dependencies
import math
import numpy as np
import matplotlib.pyplot as plt
import time

def ldc2D(Re=100,N=101, Num_ts = 50000, omega = 1.8, visTs=10000):
    # density and kinematic viscosity of glycol
    rho_p = 965.3 # kg/m^3
    nu_p = 0.06/rho_p
    
    # set Geometry
    Lx = 1.0 # cavity length in the x-direction
    Ly = 1.0 # cavity length in the y-direction
    # Based on the Reynolds number, characteristic length and fluid viscosity,
    # compute the corresponding lid velocity:
    
    L_o = Ly # characteristic length
    V_o = Re*nu_p/L_o # corresponding characteristic velocity
    T_o = L_o/V_o
    
    # convert to dimensionless units
    T_d = 1.0
    L_d = 1.0
    U_d = (T_o/L_o)*V_o
    nu_d = 1./float(Re)
    
    # convert to LBM units
    dx = 1./(float(N) - 1.)
    dt = (dx**2.)*float(Re)*((1./3.)*((1./omega)-0.5))
    u_lbm = (dt/dx)*U_d
    
    # get conversion factors from lattice units to physical units
    u_conv_fact = (dx/dt)*(L_o/T_o) #multiply LBM velocity by u_conv_fact to get physical velocity 
    t_conv_fact = (dt*T_o) # multiply LBM time step by t_conv_fact to get physical time
    l_conv_fact = dx*L_o # multiply LBM lattice location by l_conv_fact to get physical distance
    p_conv_fact = ((l_conv_fact/t_conv_fact)**2.)*(1./3.) # multiply LBM density by p_conv_fact to get pressure
    
    rho_lbm = rho_p # not prescribed, but I have never had a problem with doing it this way.
    
    print(f'Lid velocity = %5.4f m/sec' % V_o);
    print(f'LBM flow Mach number = %5.4f' % u_lbm);
    print(f'Physical time simulated = %6.3f seconds' % (t_conv_fact*(float(Num_ts))))
    
    numSpd = 9
    Ny = int(math.ceil((Ly/L_o)*(float(N))))
    Nx = int(math.ceil((Lx/L_o)*float(N)))
    nnodes = Nx*Ny
    
    # need to identify which lattice points are on stationary and moving boundaries - to do so, we
    # define the geometry:
    x_left = 0.; x_right = x_left + Lx
    y_bottom = 0.; y_top = y_bottom + Ly
    x_space = np.linspace(x_left,x_right,Nx,dtype=np.float64)
    y_space = np.linspace(y_bottom,y_top,Ny,dtype=np.float64)
    xx,yy = np.meshgrid(x_space,y_space)
    gcoord = np.zeros((2,nnodes))
    gcoord[0][:] = np.reshape(xx,nnodes) # x-coordinate values
    gcoord[1][:] = np.reshape(yy,nnodes) # y-coordinate values
    
    # find which node numbers are along the line y == 0
    bottom_nodes = np.argwhere(gcoord[1][:]<dx/2.).flatten()
    left_nodes = np.argwhere(gcoord[0][:]<dx/2.).flatten()
    right_nodes = np.argwhere(gcoord[0][:]>(Lx - dx/2.)).flatten()
    moving_nodes = np.argwhere(gcoord[1][:]>(Ly - dx/2.)).flatten()
    solid_nodes = np.unique(np.concatenate((bottom_nodes,left_nodes,right_nodes)))
    
    # removes any previously identified solid nodes from the moving node list.
    moving_nodes = np.setxor1d(moving_nodes,np.intersect1d(moving_nodes,solid_nodes))
    
    
    # check against other code
    node_type = np.zeros(nnodes).astype(np.int32)
    node_type[solid_nodes]=1
    node_type[moving_nodes]=2
    
    #print str(node_type)

    # initialize data structures to hold the density distribution functions
    fIn = np.ones((numSpd,nnodes),dtype=np.float64)
    w = np.array([[4./9.], [1./9.],[1./9.],[1./9.],[1./9.],
        [1./36.],[1./36.],[1./36.],[1./36.]])
    for spd in range(numSpd):
        fIn[spd][:]*=rho_lbm*w[spd]
        
    fOut = np.copy(fIn)
    
    
    ex = np.array([0.,1.,0.,-1.,0.,1.,-1.,-1.,1.])
    ey = np.array([0.,0.,1.,0.,-1.,1.,1.,-1.,-1.])
    bb_spd = [0,3,4,1,2,7,8,5,6];
    fEq = np.zeros_like(fIn)
    snl = solid_nodes
    vnl = moving_nodes
    u = u_lbm
    
    # make a stream target matrix - incorporate periodicity to the lattice here.
    stm = np.zeros((numSpd,nnodes),dtype=np.int)
    
    ind = np.arange(nnodes)
    ind = np.reshape(ind,(Ny,Nx))
    for spd in range(numSpd):
        tInd = np.roll(ind,-int(ex[spd]),axis=1)
        tInd = np.roll(tInd,-int(ey[spd]),axis=0)
        tInd = np.reshape(tInd,(1,nnodes))
        stm[spd][:] = tInd
    # commence time stepping
    t0 = time.time()
    for t in range(Num_ts):
        if t%50 == 0:
            print(f'Commencing time step %i' % t)
           
        # compute density
        rho = np.sum(fIn,axis=0)
        
        # compute velocity
        ux = np.dot(np.transpose(ex),fIn)/rho
        uy = np.dot(np.transpose(ey),fIn)/rho
        
        # set microscopic Dirichlet-type BC
        ux_t = ux[vnl]; uy_t = uy[vnl];
        dx = u - ux_t; dy = 0. - uy_t;
        for spd in range(1,numSpd):
            cu = 3.*(ex[spd]*dx + ey[spd]*dy)
            fIn[spd][vnl]+=w[spd]*(rho[vnl]*cu)
               
        # set macroscopic Dirichlet-type boundary conditions
        ux[snl]=0.; uy[snl]=0.
        ux[vnl]=u; uy[vnl]=0.
        
        # compute Equilibrium density distribution
        fEq = np.zeros_like(fIn)
        
        for spd in range(numSpd):
            cu = 3.*(ex[spd]*ux + ey[spd]*uy)
            fEq[spd][:]=w[spd]*rho*(1.+cu + (0.5)*cu*cu -
                                    (3./2.)*(ux*ux + uy*uy))
            
        # collide
        fOut = fIn - (fIn - fEq)*omega
        
        # bounce-back the solid nodes 
        for spd in range(numSpd):
            fOut[spd][snl]=fIn[bb_spd[spd]][snl]
            
        # stream
        for spd in range(numSpd):
            fIn[spd][stm[spd][:]]=fOut[spd][:]
        
        
        # write fIn data out after time step
        #np.save('step1_gold',fIn)
            
        if t%visTs==0 and t>0:
            # do some visualization of the output
            uMag = np.sqrt(ux*ux + uy*uy)
            fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(6,10))
            ax1.imshow(np.reshape(uMag,(Ny,Nx)),extent=[0,Nx,0,Ny])
            ax1.set_title('Velocity')
            pressure = rho*p_conv_fact
            pressure -= pressure[int(nnodes/2)] # make pressure relative
            ax2.imshow(np.reshape(pressure,(Ny,Nx)),extent=[0,Nx,0,Ny])
            ax2.set_title('Density')
            plt.tight_layout()
            plt.show()
            
        
    t1 = time.time();
    elapsedTime = t1 - t0;
    LPU = float(nnodes)*float(Num_ts)
    LPU_sec = LPU/float(elapsedTime)
    print(f'Lattice point updates per second = %g.' % LPU_sec)
    
    uMag = np.sqrt(ux*ux + uy*uy)
    fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(6,10))
    ax1.imshow(np.reshape(uMag,(Ny,Nx)),extent=[0,Nx,0,Ny])
    ax1.set_title('Velocity')
    ax2.imshow(np.reshape(rho,(Ny,Nx)),extent=[0,Nx,0,Ny],cmap='gist_rainbow')
    ax2.set_title('Density')
    plt.tight_layout()
    plt.show()

ldc2D(100,101,20000,1.8,10000)
#ldc2D(10,4,1)
