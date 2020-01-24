# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from pbpl.common.units import *

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Create CST initial conditions for trajectory map',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-compton-calc-map-particles 0.240 24.0 80 -1.0 40.01 1.0 --output=18000A.pid
''')
    parser.add_argument(
        '--output', metavar='output-file', default='particles.pid',
        help='Output file (default=particles.pid)')
    parser.add_argument(
        'E0', metavar='FLOAT', type=float, help='E0 (MeV)')
    parser.add_argument(
        'E1', metavar='FLOAT', type=float, help='E1 (MeV)')
    parser.add_argument(
        'num_E', metavar='INT', type=float, help='num energy log bins')
    parser.add_argument(
        'y0', metavar='FLOAT', type=float, help='y0 (mm)')
    parser.add_argument(
        'y1', metavar='FLOAT', type=float, help='y1 (mm)')
    parser.add_argument(
        'dy', metavar='INT', type=float, help='dy (mm)')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    m0 = me
    rest_energy = m0*c_light**2
    x0 = 0*mm
    z0 = 0*mm

    energy_vals = np.logspace(
        np.log10(args.E0), np.log10(args.E1), args.num_E)*MeV
    y0_vals = np.arange(args.y0, args.y1, args.dy)*mm

    f = open(args.output, 'w')
    for kinetic_energy in energy_vals:
        for y0 in y0_vals:
            for q0 in [-eplus]:
                for theta in [0.0]:
                    for phi in [0.0]:
                        total_energy = rest_energy + kinetic_energy
                        gamma0 = total_energy/rest_energy
                        beta0 = np.sqrt(1-1/gamma0**2)
                        p0 = gamma0*m0*c_light*beta0
                        n0 = np.array(
                            (np.cos(phi)*np.sin(theta),
                             np.sin(phi)*np.sin(theta), np.cos(theta)))
                        r0 = np.array((x0, y0, z0))
                        beta0_gamma0 = n0 * beta0 * gamma0
                        current = 1e-15*amp
                        fmt = (
                            '{:12.5e} {:12.5e} {:12.5e} ' +
                            '{:12.5e} {:12.5e} {:12.5e} ' +
                            '{:12.5e} {:12.5e} {:12.5e}\n')
                        f.write(
                            fmt.format(*r0/meter, *beta0_gamma0,
                                       m0/kg, q0/coulomb, current/amp))
    f.close()

if __name__ == '__main__':
    sys.exit(main())
