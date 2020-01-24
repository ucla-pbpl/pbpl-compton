#!/usr/bin/env python
import Geant4 as g4

def repeater(particle, energy, x0, direction):
    x0 = g4.G4ThreeVector(*x0)
    direction = g4.G4ThreeVector(*direction)
    while 1:
        yield particle, x0, direction, energy
