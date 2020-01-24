#!/usr/bin/env python
import sys
import argparse
import numpy as np
from pbpl.common.units import *
import h5py
from scipy.linalg import norm
from collections import namedtuple

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Convert CST trajectories ASCII to HDF5',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-convert-trajectories
''')
    parser.add_argument(
        '--output', metavar='output-file', default='trajectories.h5',
        help='Output file (default=trajectories.h5)')
    parser.add_argument(
        'input', metavar='input-file',
        help='Input file')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

def dump_dxf():
    A = np.loadtxt(
        'trajectories.txt', usecols=(0,2,10), dtype=np.float, skiprows=7)
    dwg = ezdxf.new('R2000')
    msp = dwg.modelspace()
    for i in range(int(A[:,2].max())+1):
        mask = A[:,2] == i
        x = A[mask,0]*meter
        y = A[mask,1]*meter
        msp.add_lwpolyline(np.array((x/mm, y/mm)).T)
    dwg.saveas('trajectories.dxf')

Trajectory = namedtuple('Trajectory', 'm0 q0 x p t')

def load_trajectories(filename):
    # input col  A col  variable
    # =========  =====  ========
    # 0,1,2      0,1,2  x,y,z
    # 3,4,5      3,4,5  px,py,pz
    # 6,7        6,7    m0,q0
    # 9          8      time
    # 10         9      id

    A = np.loadtxt(
        filename, usecols=(0,1,2,3,4,5,6,7,9,10),
        dtype=np.float32, skiprows=7)

    result = []
    for i in range(int(A[:,-1].max())+1):
        mask = A[:,-1] == i
        first_index = mask.argmax()
        m0 = A[first_index, 6]*kg
        # q0 = A[first_index, 7]*coulomb
        # CST writes some goofy value of particle charge.  Override for now...
        q0 = -eplus
        x = A[mask, 0:3]*meter
        p = A[mask, 3:6]*m0*c_light
        t = A[mask, 8]*ns
        p0 = norm(p[0])
        E0 = np.sqrt(p0**2*c_light**2 + m0**2*c_light**4)
        K0 = E0 - m0*c_light**2
        result.append(
            Trajectory(m0/kg, q0/coulomb, x/meter, p/(m0*c_light), t/sec))
    return result

def dump_namedtuple(gout, nt, compress=False):
    for k, v in nt._asdict().items():
        if compress and isinstance(v, np.ndarray):
            gout.create_dataset(k, data=v, compression='gzip')
        else:
            gout[k] = v

def save_trajectories(fout, trajectories):
    for i in range(len(trajectories)):
        gout = fout.create_group(str(i))
        dump_namedtuple(gout, trajectories[i])

def main():
    args = get_args()
    trajectories = load_trajectories(args.input)
    with h5py.File(args.output, 'w') as fout:
        save_trajectories(fout, trajectories)

if __name__ == '__main__':
    sys.exit(main())
