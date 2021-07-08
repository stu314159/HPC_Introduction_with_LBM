#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:38:57 2021

@author: sblair
"""

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

your_mesh = mesh.Mesh.from_file('HalfDonut.stl')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale,scale,scale)

pyplot.show()
