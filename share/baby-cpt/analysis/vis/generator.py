#!/usr/bin/env python
import Geant4 as g4
from Geant4.hepunit import *
import numpy as np
from numpy import cos, sin

def ugh():
    yield 'e-', g4.G4ThreeVector(), g4.G4ThreeVector(0,0,1), 4*MeV

# current (kA)  E0 (keV)  E1 (MeV)   z0 (mm)   z1 (mm)
# ============  ========  ========   =======   =======
# 18            280       23.2       7.3       218.8
# 9             119       14.2       7.1       219.1
# 4.5           32        6.9        7.0       219.7
def pattern_spray():
    energies = (14.2*MeV)/2**np.arange(7)
    # energies = np.array((30*keV,))
    for particle in ['e+', 'e-']:
        for energy in energies:
            yield particle, g4.G4ThreeVector(), g4.G4ThreeVector(0,0,1), energy

    # energies = (14.2*MeV/2)/2**np.arange(6)
    # for particle in ['e-']:
    #     for energy in energies:
    #         yield particle, g4.G4ThreeVector(0,-10*mm,0), g4.G4ThreeVector(0,0,1), energy

    # energies = (14.2*MeV/4)/2**np.arange(5)
    # for particle in ['e-']:
    #     for energy in energies:
    #         yield particle, g4.G4ThreeVector(0,-20*mm,0), g4.G4ThreeVector(0,0,1), energy

    energies = (14.2*MeV/8)/2**np.arange(4)
    for particle in ['e-']:
        for energy in energies:
            yield particle, g4.G4ThreeVector(0,-30*mm,0), g4.G4ThreeVector(0,0,1), energy

def collimator_spray():
    direction = g4.G4ThreeVector(0,0,1)
    for particle in ['e+', 'e-']:
        for energy in np.logspace(np.log10(0.100), np.log10(3.0), 4)*MeV:
            for y0 in np.arange(0, -20.01, -10)*mm:
                location = g4.G4ThreeVector(0,y0,0)
                yield particle, location, direction, energy

def random_collimator_spray(theta=0*deg):
    while 1:
        energy = MeV*10**np.random.uniform(np.log10(0.240), np.log10(16.0))
        y0 = np.random.uniform(-20.0, 20.0)*mm
        location = g4.G4ThreeVector(0,y0,0)
        phi = np.random.uniform(0, 2*pi)
        direction = g4.G4ThreeVector(
            cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta))
        yield 'e-', location, direction, energy

def repetitive_spray(particle, energy, x0, y0, z0):
    direction = g4.G4ThreeVector(0,0,1)
    while 1:
        yield particle, g4.G4ThreeVector(x0, y0, z0), direction, energy
