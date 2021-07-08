#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:37:08 2021

@author: sblair
"""

import sys
sys.path.insert(1,'.')

import FluidChannel as fc

aLx_p = 4; aLy_p = 4; aLz_p = 9.0;

stl_filename = 'HalfDonut_moved.stl';

myObst = fc.STL_Obstruction(stl_filename);

myChan = fc.FluidChannel(Lx_p=aLx_p,Ly_p=aLy_p,Lz_p=aLz_p,
                         obst=myObst,N_divs=50);

myChan.write_bc_vtk();