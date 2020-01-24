#!/usr/bin/env python
import sys
import copy
import toml
import numpy as np
from pbpl import compton
from Geant4.hepunit import *
import common

def reconf(conf, desc, num_events, theta0):
    result = copy.deepcopy(conf)
    result['PrimaryGenerator']['NumEvents'] = num_events
    result['PrimaryGenerator']['PythonGeneratorArgs'] = [theta0]
    result['Detectors']['ComptonScint']['File'] = 'out/' + desc + '.h5'
    return result

def main():
    tr = compton.ParallelTaskRunner()
    conf = toml.load('pwfa.toml')

    for theta0 in common.emission_angles:
        desc = '{:02d}deg'.format(int(round(theta0/deg)))
        tr.add_task(compton.Task(
            reconf(conf, desc, common.num_events, theta0),
            desc, 'pbpl-compton-mc'))
    tr.run()

if __name__ == '__main__':
    sys.exit(main())
