# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import h5py

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Combine multiple deposition files via summation',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-compton-sum-deposition A.h5 B.h5 C.h5 out.h5
''')
    parser.add_argument(
        'infiles', metavar='INFILE',
        nargs='+', help='Input HDF5 deposition file')
    parser.add_argument(
        'outfile', metavar='OUTFILE',
        help='Output HDF5 deposition file')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # - Any dataset named 'edep' or 'num_events' is summed in the output.
    # - Otherwise, datasets are simply copied to the output.  These
    #   datasets must be identical in each input file.
    with h5py.File(args.outfile, 'w') as fout:
        for filename in args.infiles:
            with h5py.File(filename, 'r') as fin:
                def visit(k, v):
                    if not isinstance(v, h5py.Dataset):
                        return
                    if k not in fout:
                        fin.copy(v, fout, k)
                    else:
                        if k.split('/')[-1] in ['edep', 'num_events']:
                            fout[k][()] += v[()]
                        else:
                            assert(np.array_equal(fout[k][()], v[()]))
                fin.visititems(visit)
