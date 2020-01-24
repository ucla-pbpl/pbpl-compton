#!/usr/bin/env python
import sys
import copy
import toml
import numpy as np
from pbpl import compton
from Geant4.hepunit import *
import random
import string

def reconf(conf, desc, func_index, num_events):
    result = copy.deepcopy(conf)
    result['Detectors']['ComptonScint']['File'] = 'out/' + desc + '.h5'
    result['PrimaryGenerator']['NumEvents'] = num_events
    result['PrimaryGenerator']['PythonGeneratorArgs'] = [
        '{}'.format(num_events), '\'{}\''.format(desc)]
    return result

def main():
    num_events = 20000000#2e6
    random_affix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    tr = compton.ParallelTaskRunner()
    conf = toml.load('pwfa.toml')
    for f_i in range(0, 10):
        desc = 'gYrE-col-2e7-{}-{}'.format(random_affix, f_i)
        tr.add_task(compton.Task(
            reconf(
                conf, desc, f_i, num_events),
            desc, 'pbpl-compton-mc'))
    tr.run()

if __name__ == '__main__':
    sys.exit(main())
