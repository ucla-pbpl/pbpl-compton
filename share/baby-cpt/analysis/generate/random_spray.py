import Geant4 as g4
from Geant4.hepunit import *
import numpy as np
import scipy.stats as stats
import random
import generate_distrib

def get_energies(num):
    bins=50
    lower, upper = 0.25, 25
    weights = np.zeros([ bins])
    #1. random distrib
    for x in range(1, bins):
       weights[x]=random.randint(0, 99)
    #2. random mono
    #weights[random.randint(0, bins-1)]=1
    energies = np.linspace(lower, upper, bins)
    energy = random.choices(energies, weights, k=num)
    #3. random centered gaussian
    #mean = random.uniform(lower, upper)
    #sigma = (upper-lower)/7
    #X = stats.truncnorm(
    #(lower - mean) / sigma, (upper - mean) / sigma, loc=mean, scale=sigma)
    #energy = X.rvs(num)
    #4. other stats function
    #energy = generate_distrib.generate_distrib(lower, upper, stats.rayleigh, num)
    return energy

def get_ys(num):
    y_bins=50
    lower = -29
    upper = 30
    weights = np.zeros([ y_bins])
    #1. random distrib
    #for x in range(0, y_bins):
    #    weights[x]=random.randint(0, 99)
    #2. random mono
    #weights[random.randint(0, y_bins-1)]=1
    #ys = np.linspace(lower, upper, y_bins)
    #y = random.choices(ys, weights, k=num)
    #3. random centered gaussian
    mean = random.uniform(lower, upper)
    sigma = (upper-lower)/7
    X = stats.truncnorm(
    (lower - mean) / sigma, (upper - mean) / sigma, loc=mean, scale=sigma)
    y = X.rvs(num)
    #4. other stats function
    #y = generate_distrib.generate_distrib(lower, upper, stats.cosine, num, [10])
    return y

def gamma_spray(total, desc):
    ys = get_ys(total)
    i=0
    energies = get_energies(total)
    histo, _, _ = np.histogram2d(ys, energies, range=[[-29, 30],[0, 25]], bins=[50, 50])
    #print("here once")
    np.savez("{}.npz".format(desc), histo=histo)
    while i<total:
        yield 'gamma', g4.G4ThreeVector(0, ys[i]*mm,-25*mm), g4.G4ThreeVector(0,0,1), energies[i]
        i=i+1
