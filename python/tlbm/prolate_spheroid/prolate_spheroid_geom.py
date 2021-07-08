#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:33:09 2021

@author: sblair
"""

import sys
sys.path.insert(0,'.');

import FluidChannel as fc
import numpy as np

aLx_p = 0.41;
aLy_p = 0.41;
aLz_p = 1.5;

aNdivs = 25;

proSph = fc.FluidChannel(Lx_p=aLx_p,Ly_p=aLy_p,
                         Lz_p=aLz_p,
                         fluid='glycol',
                         obst=fc.ProlateSpheroid(x_c=aLx_p/2.,y_c=aLy_p/2.,
                                                 z_c=aLz_p/2.,ab=0.032,
                                                 c=0.216,aoa=np.pi/8.),
                         N_divs=aNdivs);

proSph.set_pRef_indx(aLx_p/2., aLy_p/2., 0.96*aLz_p);

proSph.write_mat_file('prolateSpheroid');
proSph.write_bc_vtk();