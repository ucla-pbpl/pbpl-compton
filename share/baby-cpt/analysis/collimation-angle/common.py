import numpy as np
from Geant4.hepunit import *

num_events = 2000000
lim_E = np.array((240*keV, 16*MeV))
lim_y0 = np.array((-20*mm, 20*mm))
emission_angles = np.arange(0, 10.1, 2)*deg
