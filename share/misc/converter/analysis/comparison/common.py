import numpy as np
from Geant4.hepunit import *

num_events = 50000000
thickness_vals = (0.25*mm) * 2**np.arange(6)
# thickness_vals = (0.25*mm) * 2**np.arange(6, None, 2)
# energy_vals = (0.25*MeV) * 2**np.arange(6, None, 2)
energy_vals = np.array((0.25*MeV,2.5*MeV,25*MeV))
