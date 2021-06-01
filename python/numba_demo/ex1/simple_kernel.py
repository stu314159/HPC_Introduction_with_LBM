#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:37:15 2021

@author: sblair
"""

from numba import cuda

@cuda.jit
def add_kernel(x,y,out):
    tx = cuda.threadIdx.x;
    ty = cuda.threadIdx.y;
    
    block_size = cuda.blockDim.x;
    grid_size = cuda.gridDim.x;
    
    start = tx + ty * block_size;
    stride = block_size * grid_size;
    
    for i in range(start,x.shape[0],stride):
        out[i] = x[i] + y[i];
        

import numpy as np

n = 5000000;
x = np.arange(n).astype(np.float32);
y = 2*x;
out = np.empty_like(x);

threads_per_block = 128;
blocks_per_grid = 30;

add_kernel[blocks_per_grid,threads_per_block](x,y,out);
print(out[:10])
