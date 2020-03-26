import Geant4 as g4
from Geant4.hepunit import *
import numpy as np
import scipy.stats as stats
import random
import generate_distrib
import image_spray
import triangle_spray

def get_distrib_random(num, bins, lower, upper):
    weights = np.zeros([ bins])
    #1. random distrib
    for x in range(1, bins):
       weights[x]=random.randint(0, 99)
    energies = np.linspace(lower, upper, bins)
    energy = random.choices(energies, weights, k=num)
    return energy

def get_distrib_mono(num, bins, lower, upper):
    weights = np.zeros([ bins])
    #2. random mono
    weights[random.randint(0, bins-1)]=1
    energies = np.linspace(lower, upper, bins)
    energy = random.choices(energies, weights, k=num)
    return energy

def get_distrib_gaussian(num, bins, lower, upper):
    #3. random centered gaussian
    mean = random.uniform(lower, upper)
    sigma = (upper-lower)/7
    X = stats.truncnorm(
    (lower - mean) / sigma, (upper - mean) / sigma, loc=mean, scale=sigma)
    energy = X.rvs(num)
    return energy

def get_distrib_rayleigh(num, bins, lower, upper):
    #4. other stats function
    energy = generate_distrib.generate_distrib(lower, upper, stats.rayleigh, num)
    return energy

def get_distrib_cosine(num, bins, lower, upper):
    y = generate_distrib.generate_distrib(lower, upper, stats.cosine, num, [10])
    return y


def gamma_spray(total, desc, y_bins, y_lower, y_upper, e_bins, e_lower, e_upper):
    #first training
    #y_bins=50
    #y_lower = -29
    #y_upper = 30
    #e_bins=50
    #e_lower, e_upper = 0.25, 25

    i=0
    ln_e_lower = np.log(e_lower)
    ln_e_upper = np.log(e_upper)

    get_distrib_func = {
        "r": get_distrib_random,
        "g": get_distrib_gaussian,
        "c": get_distrib_cosine,
        "a": get_distrib_rayleigh,
        "m": get_distrib_mono,
    }

    tags = desc.split("-")
    tag = tags[0]
    #gYrE-col-2e7-KBYQ19-7
    if(tag == "triag"):
        ye = triangle_spray.triangle_spray(y_bins, y_lower, y_upper, e_bins, ln_e_lower, ln_e_upper, total)
        histo, _, _ = np.histogram2d(ye[:, 0], ye[:, 1], 
            range=[[y_lower, y_upper],[ln_e_lower, ln_e_upper]], 
            bins=[y_bins, e_bins])
    elif (tag == "image"):
        ye = image_spray.image_spray(y_bins, y_lower, y_upper, e_bins, ln_e_lower, ln_e_upper, total)
        histo, _, _ = np.histogram2d(ye[:, 0], ye[:, 1], 
            range=[[y_lower, y_upper],[ln_e_lower, ln_e_upper]], 
            bins=[y_bins, e_bins])
    else:
        Y_command_index = tag.find("Y")-1
        E_command_index = tag.find("E")-1
        if (len(tag) != 4 or Y_command_index<0 or E_command_index<0 ):
            print ("Illegal tag")
            return
        else:
            yc = tag[Y_command_index]
            ec = tag[E_command_index]
            ys = get_distrib_func[yc](total, y_bins, y_lower, y_upper)
            es = get_distrib_func[ec](total, e_bins, ln_e_lower, ln_e_upper)
            histo, _, _ = np.histogram2d(ys, es, 
                range=[[y_lower, y_upper],[ln_e_lower, ln_e_upper]], 
                bins=[y_bins, e_bins])
            ye = np.column_stack((ys, es))

    while i<total:
        yield 'gamma', g4.G4ThreeVector(0, ye[i, 0]*mm,-25*mm), g4.G4ThreeVector(0,0,1), np.exp(ye[i, 1])
        i=i+1
