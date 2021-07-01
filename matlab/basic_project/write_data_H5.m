function []=write_data_H5(filename,nnodes,u,v,w,umag,p)

% build the HDF5 structure
h5create(filename,'/velo_group/x_velo',[nnodes,1],'Datatype','double');
h5create(filename,'/velo_group/y_velo',[nnodes,1],'Datatype','double');
h5create(filename,'/velo_group/z_velo',[nnodes,1],'Datatype','double');
h5create(filename,'/velo_group/velmag',[nnodes,1],'Datatype','double');
h5create(filename,'/pres_group/presmag',[nnodes,1],'Datatype','double');

% write the data
h5write(filename,'/velo_group/x_velo',u);
h5write(filename,'/velo_group/y_velo',v);
h5write(filename,'/velo_group/z_velo',w);
h5write(filename,'/velo_group/velmag',umag);
h5write(filename,'/pres_group/presmag',p);