#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 07:51:51 2021

@author: sblair
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank();
size = comm.Get_size();

print 'Hello from process %d of %d!!' % (rank, size);
