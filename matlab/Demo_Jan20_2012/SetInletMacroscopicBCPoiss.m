% appropriate for D2Q9 grids only

function [ux,uy,rho]=SetInletMacroscopicBCPoiss(ux,uy,rho,fIn,u_bc,b,...
    gcoord,inlet_node_list)

 % set macroscopic inlet boundary conditions
    ux(inlet_node_list) = (u_bc)*(1-((gcoord(inlet_node_list,2)-b).^2)./(b*b));
    uy(inlet_node_list) = 0;
    rho(inlet_node_list) = 1./(1-ux(inlet_node_list)).*(...
        fIn(inlet_node_list,1)+fIn(inlet_node_list,3)+fIn(inlet_node_list,5)...
        +2*(fIn(inlet_node_list,4)+fIn(inlet_node_list,7)+...
        fIn(inlet_node_list,8)));