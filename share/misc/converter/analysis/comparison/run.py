#!/usr/bin/env python
import sys
import copy
import toml
import numpy as np
from pbpl import compton
from Geant4.hepunit import *
import common

def reconf(conf, desc, num_events, thickness, energy):
    result = copy.deepcopy(conf)
    result['PrimaryGenerator']['NumEvents'] = num_events
    result['PrimaryGenerator']['PythonGeneratorArgs'][1] = (
        '{}*MeV'.format(energy/MeV))
    A = result['Geometry']['World']['Converter']
    A['pZ'] = float(0.5*thickness)
    A['Transformation'][1][0] = float(-0.5*thickness)
    result['Detectors']['Converter']['File'] = 'out/' + desc + '.h5'
    return result

def main():
    tr = compton.ParallelTaskRunner()
    conf = toml.load('converter.toml')

    for thickness in common.thickness_vals:
        for energy in common.energy_vals:
            desc = '{:.3f}mm_{:.3f}MeV'.format(
                round(thickness/mm, 3), round(energy/MeV, 3))
            tr.add_task(compton.Task(
                reconf(conf, desc, common.num_events, thickness, energy),
                desc, 'pbpl-compton-mc'))
    tr.run()

if __name__ == '__main__':
    sys.exit(main())
