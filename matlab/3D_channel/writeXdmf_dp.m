function []=writeXdmf_dp(dims,dx,filename,h5_filename)

fid = fopen(filename,'w');
fprintf(fid,'<?xml version="1.0" ?>\n');
fprintf(fid,'<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n');
fprintf(fid,'<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.1">\n');
fprintf(fid,'<Domain>\n');

fprintf(fid,'<Grid Name="my_Grid" GridType="Uniform">\n');
fprintf(fid,'<Topology TopologyType="3DCoRectMesh" Dimensions="%d %d %d">\n',...
    dims(1),dims(2),dims(3));
fprintf(fid,'</Topology>\n');

fprintf(fid,'<Geometry GeometryType="Origin_DxDyDz">\n');
fprintf(fid,'<DataItem Dimensions="3" NumberType="Integer" Format="XML">\n');
fprintf(fid,'0 0 0\n') ;
fprintf(fid,'</DataItem>\n');
fprintf(fid,'<DataItem Dimensions="3" NumberType="Integer" Format="XML">\n');

fprintf(fid,'%g %g %g \n',dx,dx,dx);
fprintf(fid,'</DataItem>\n');
fprintf(fid,'</Geometry>\n');

fprintf(fid,'<Attribute Name="velocity" AttributeType="Vector" Center="Node">\n');
fprintf(fid,...
    '<DataItem ItemType="Function" Function="JOIN($0, $1, $2)" Dimensions="%d %d %d 3">\n',...
    dims(1),dims(2),dims(3));
fprintf(fid,...
    '<DataItem Dimensions="%d %d %d" NumberType="Double" Format="HDF">\n',...
    dims(1),dims(2),dims(3));

fprintf(fid,'%s:/velo_group/x_velo\n',h5_filename);
fprintf(fid,'</DataItem>\n');
fprintf(fid,...
    '<DataItem Dimensions="%d %d %d" NumberType="Double" Format="HDF">\n',...
    dims(1),dims(2),dims(3));

fprintf(fid,'%s:/velo_group/y_velo\n',h5_filename);
fprintf(fid,'</DataItem>\n');
fprintf(fid,...
    '<DataItem Dimensions="%d %d %d" NumberType="Double" Format="HDF">\n',...
    dims(1),dims(2),dims(3));

fprintf(fid,'%s:/velo_group/z_velo\n',h5_filename);
fprintf(fid,'</DataItem>\n');
fprintf(fid,'</DataItem>\n');
fprintf(fid,'</Attribute>\n');
fprintf(fid,'<Attribute Name="pressure" AttributeType="Scalar" Center="Node">\n');
fprintf(fid,...
    '<DataItem Dimensions="%d %d %d" NumberType="Double" Format="HDF">\n',...
    dims(1),dims(2),dims(3));

fprintf(fid,'%s:/pres_group/presmag\n',h5_filename);
fprintf(fid,'</DataItem>\n');
fprintf(fid,'</Attribute>\n');
fprintf(fid,'<Attribute Name="velocityMagnitude" AttributeTpe="Scalar" Center="Node">\n');
fprintf(fid,...
    '<DataItem Dimensions="%d %d %d" NumberType="Double" Format="HDF">\n',...
    dims(1),dims(2),dims(3));

fprintf(fid,'%s:/velo_group/velmag\n',h5_filename);
fprintf(fid,'</DataItem>\n');
fprintf(fid,'</Attribute>\n');

fprintf(fid,'</Grid>\n');
fprintf(fid,'</Domain>\n');
fprintf(fid,'</Xdmf>\n');


fclose(fid);