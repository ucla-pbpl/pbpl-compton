#!/usr/bin/env python
import sys
import copy
import toml
import numpy as np
from pbpl import compton
from Geant4.hepunit import *

# current (kA)  z (mm)  E (MeV)
# ============  ======  =======
# 4.5           0       0.023
# 4.5           5       0.029
# 4.5           10      0.036
# 4.5           220     6.926
# 4.5           225     7.408
# 4.5           230     7.917

# 9             0       0.089
# 9             5       0.109
# 9             10      0.133
# 9             220     14.375
# 9             225     15.429
# 9             230     16.565

# 18            0       0.216
# 18            5       0.258
# 18            10      0.307
# 18            215     22.010
# 18            220     23.606
# 18            225     25.321
# 18            230     27.165

# z0=5mm  and z1=225mm

def reconf(conf, current, theta0, num_events, filename):
    result = copy.deepcopy(conf)
    result['PrimaryGenerator']['NumEvents'] = int(num_events)
    if current == '4500':
        E_lim = [30*keV, 7.4*MeV]
    elif current == '9000':
        E_lim = [100*keV, 15.4*MeV]
    elif current == '18000':
        E_lim = [250*keV, 25.3*MeV]
    else:
        assert(False)
    result['PrimaryGenerator']['PythonGeneratorArgs'][3] = (
        '[MeV*{}, MeV*{}]'.format(*np.array(E_lim)/MeV))
    result['PrimaryGenerator']['PythonGeneratorArgs'][2] = (
        '[deg*{0}, deg*{0}]'.format(theta0/deg))
    result['Detectors']['ComptonScint']['File'] = filename
    result['Fields']['Cpt']['File'] = '../../field/B-field-{}A.h5'.format(
        current)
    return result

def main():
    indices = [
        [['4500', '9000', '18000'], 'current', None],
        [deg*np.array((0, 2, 4, 8)), 'theta0', 'deg']]
    num_events_per_run = 1000000
    min_events_per_thread = 10000
    max_num_threads = 100

    conf = toml.load('baby-cpt.toml')
    compton.RunMonteCarlo(
        indices, conf, reconf, 'out/12mm-2mm.h5',
        num_events_per_run, min_events_per_thread, max_num_threads)

    # del conf['Geometry']['World']['Collimator']
    # compton.RunMonteCarlo(
    #     indices, conf, reconf, 'out/uncollimated.h5',
    #     num_events_per_run, min_events_per_thread, max_num_threads)


if __name__ == '__main__':
    sys.exit(main())
