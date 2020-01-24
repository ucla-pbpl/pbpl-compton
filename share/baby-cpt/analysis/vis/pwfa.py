#!/usr/bin/env python
import Geant4 as g4
from Geant4.hepunit import *
import numpy as np
from numpy import cos, sin

def pattern_spray():
    energies = (15*MeV)/2**np.arange(7)
    for particle in ['e+', 'e-']:
        for energy in energies:
            yield particle, g4.G4ThreeVector(), g4.G4ThreeVector(0,0,1), energy

    energies = (15*MeV/8)/2**np.arange(3)
    for particle in ['e-']:
        for energy in energies:
            yield particle, g4.G4ThreeVector(0,-30*mm,0), g4.G4ThreeVector(0,0,1), energy

def collimator_spray():
    direction = g4.G4ThreeVector(0,0,1)
    for particle in ['e+', 'e-']:
        for energy in np.logspace(np.log10(0.240), np.log10(14.0), 8)*MeV:
            for y0 in np.linspace(-40.0, 40.0, 7)*mm:
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
