#!/usr/bin/env python
import Geant4 as g4
from Geant4.hepunit import *
import numpy as np
from numpy import cos, sin
import common

def random_spray(theta0):
    while 1:
        energy = MeV*10**np.random.uniform(
            np.log10(common.lim_E[0]/MeV), np.log10(common.lim_E[1]/MeV))
        y0 = np.random.uniform(*common.lim_y0)
        location = g4.G4ThreeVector(0,y0,0)
        phi0 = np.random.uniform(0, 2*pi)
        direction = g4.G4ThreeVector(
            cos(phi0)*sin(theta0), sin(phi0)*sin(theta0), cos(theta0))
        yield 'e-', location, direction, energy
