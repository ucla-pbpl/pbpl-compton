#!/usr/bin/env python
import sys
import argparse
import numpy as np
from pbpl.common.units import *
import h5py
from scipy.linalg import norm
from collections import namedtuple
import ezdxf

def dump_dxf():
    A = np.loadtxt(
        'trajectories-12600.txt', usecols=(0,2,10), dtype=np.float, skiprows=7)
    dwg = ezdxf.new('R2000')
    msp = dwg.modelspace()
    for i in range(int(A[:,2].max())+1):
        mask = A[:,2] == i
        x = A[mask,0]*meter
        y = A[mask,1]*meter
        msp.add_lwpolyline(np.array((x/mm, y/mm)).T)
    dwg.saveas('trajectories-12600.dxf')

dump_dxf()
