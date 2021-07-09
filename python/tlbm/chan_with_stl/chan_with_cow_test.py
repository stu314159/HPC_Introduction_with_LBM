#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 08:22:28 2021

@author: sblair
"""

import sys
sys.path.insert(1,'.')

import FluidChannel as fc

aLx_p = 10; aLy_p = 10; aLz_p = 25;

stl_filename = 'Cow_moved.stl';

myObst = fc.STL_Obstruction(stl_filename);

myChan = fc.FluidChannel(Lx_p=aLx_p,Ly_p=aLy_p,Lz_p=aLz_p,
                         obst=myObst,N_divs=121);

myChan.write_bc_vtk();