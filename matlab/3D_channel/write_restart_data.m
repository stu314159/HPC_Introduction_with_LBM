function [] = write_restart_data(u,v,w,rho,nnodes)

filename = 'restart.h5';

% if there is already a restart file; delete it
if exist(filename,'file')==2
    delete(filename);
end


% build the HDF5 structure
h5create(filename,'/velocity/x',[nnodes,1],'Datatype','double');
h5create(filename,'/velocity/y',[nnodes,1],'Datatype','double');
h5create(filename,'/velocity/z',[nnodes,1],'Datatype','double');
h5create(filename,'/density/rho',[nnodes,1],'Datatype','double');

% write the data
h5write(filename,'/velocity/x',u);
h5write(filename,'/velocity/y',v);
h5write(filename,'/velocity/z',w);
h5write(filename,'/density/rho',rho);

end

