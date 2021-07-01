#!/usr/bin/env python
import Geant4 as g4
from Geant4.hepunit import *
import numpy as np
from numpy import cos, sin

def random_spray(particle, y0_lim, theta0_lim, E_lim):
    # pick y0, theta0, and log E from flat distributions
    while 1:
        y0 = np.random.uniform(*y0_lim)
        location = g4.G4ThreeVector(0, y0, 0)
        phi0 = np.random.uniform(0, 2*pi)
        theta0 = np.random.uniform(*theta0_lim)
        direction = g4.G4ThreeVector(
            cos(phi0)*sin(theta0), sin(phi0)*sin(theta0), cos(theta0))
        energy = 10**np.random.uniform(np.log10(E_lim[0]), np.log10(E_lim[1]))
        yield particle, location, direction, energy
