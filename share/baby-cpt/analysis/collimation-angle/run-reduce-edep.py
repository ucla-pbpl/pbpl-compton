#!/usr/bin/env python
import sys
from tempfile import NamedTemporaryFile
import subprocess
import toml
import numpy as np
from pbpl import compton
from Geant4.hepunit import *
import common

def main():
    conf = {}

    conf['Output'] = {
        'Filename' : 'out/reduced-edep.h5',
        'Group' : '/'
    }

    conf['Transformation'] = [
        ['TranslateX', 'RotateY', 'TranslateZ'],
        [-40.1, -28.0, -30.0]
    ]

    conf['Input'] = {}
    for i0, theta0 in enumerate(common.emission_angles):
        filename = 'out/{:02}deg.h5'.format(int(round(theta0/deg)))
        conf['Input'][str(i0)] = filename

    # Need to convert numpy arrays to python lists because current
    # release of toml (0.10) writes numpy arrays as lists of strings
    # instead of lists of floats.  Next release of toml will include
    # a numpy-friendly encoder.
    conf['Indices'] = [
        { 'Label' : 'theta0', 'Vals' : (common.emission_angles/deg).tolist(),
          'Unit' : 'deg' },
        { 'Label': 'x', 'Unit': 'mm',
          'NumBins': 10, 'LowerEdge':  0.0, 'UpperEdge': 10.0 },
        { 'Label': 'y', 'Unit': 'mm',
          'NumBins': 60, 'LowerEdge': -75.0, 'UpperEdge': 75.0 },
        { 'Label': 'z', 'Unit': 'mm',
          'NumBins': 90, 'LowerEdge': 0.0, 'UpperEdge': 225.0 } ]

    with NamedTemporaryFile('w', delete=False) as fout:
        conf_filename = fout.name
        toml.dump(conf, fout)
        fout.close()
    proc = subprocess.call(['pbpl-compton-reduce-edep', conf_filename])

if __name__ == '__main__':
    sys.exit(main())
