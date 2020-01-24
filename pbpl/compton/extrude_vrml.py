#!/usr/bin/env python
import sys, os
import argparse
from argparse import RawDescriptionHelpFormatter
import numpy as np
import re as regex

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='Extrude VRML lines into 3D shapes.',
        epilog='Example:\n' +
        '  > pbpl-compton-extrude-vrml g4_00.wrl\n\n' +
        "Reads 'g4_00.wrl' and overwrites 'g4_00.wrl'")
    parser.add_argument(
        '--output', metavar='VRML', type=str, default=None,
        help='Specify output filename (default overwrites input)')
    parser.add_argument(
        '--radius', metavar='FLOAT', type=float, default=1.0,
        help='Cross section radius (default=1)')
    parser.add_argument(
        '--num-points', metavar='INT', type=int, default=8,
        help='Number of points in circular cross section (default=8)')
    parser.add_argument(
        'input', metavar='INPUT', type=str,
        help='Input filename (VRML format)')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.output == None:
        args.output = os.path.splitext(args.input)[0] + '.h5'
    return args

def extrude(vin, num_points, r0):
    p = regex.compile(
        r'geometry\s+IndexedLineSet\s*\{\s*coord\s*Coordinate\s*\{' +
        r'\s*point\s*\[([^]]*)[^}]*\}[^}]*\}')
    theta = np.linspace(0, 2*np.pi, num_points+1)
    xy = np.array((r0*np.cos(theta), r0*np.sin(theta)))
    cross = ''.join('{:.2f} {:.2f}, '.format(*xy) for xy in xy.T)

    def repl(m):
        result = 'geometry Extrusion {\nspine [\n' + m[1] + ']\n'
        result += 'crossSection [' + cross + ']\n'
        result += '   }\n'
        return result
    vout = p.sub(repl, vin)
    return vout

def main():
    args = get_args()
    with open(args.input, 'r') as fin:
        vin = fin.read()
    vout = extrude(vin, args.num_points, args.radius)
    with open(args.output, 'w') as fout:
        fout.write(vout)

if __name__ == '__main__':
    sys.exit(main())
