#!/usr/bin/env python
import sys
import numpy as np
import h5py
from pbpl.common.units import *

twopi = 2*np.pi

class Bins:
    def __init__(self, bins):
        self.bins = bins
        #original index numbers
        self.num_bindex = [len(x)-1 for x in bins]
        #flattened index number
        self.num_index = np.array(self.num_bindex).prod()

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr == self.num_index:
            raise StopIteration
        x = self.curr
        self.curr += 1
        return x

    def value2index(self, value):
        return self.bindex2index(self.value2bindex(value))

    def value2bindex(self, value):
        assert(len(value) == len(self.bins))
        bin_indices = []
        for v, b in zip(value, self.bins):
            i = b.searchsorted(v, side='left')
            assert(i>0 and i<len(b))
            bin_indices.append(i-1)
        return bin_indices

    def index2bindex(self, index):
        return np.unravel_index(index, self.num_bindex)

    def index2value(self, index):
        return self.bindex2value(self.index2bindex(index))

    def index2binwidth(self, index):
        return self.bindex2binwidth(self.index2bindex(index))

    def bindex2value(self, bindex):
        return np.array([
            0.5*(self.bins[i][b]+self.bins[i][b+1]) for i, b in enumerate(
                bindex)])

    def bindex2index(self, bindex):
        return np.ravel_multi_index(bindex, self.num_bindex)

    def bindex2binwidth(self, bindex):
        return np.array([
            (self.bins[i][b+1]-self.bins[i][b]) for i, b in enumerate(
                bindex)])

def load_response(fin):
    num_events = fin['num_events'][:].flatten()
    num_gamma_bins = len(num_events)
    i0 = fin['i0'][:]*MeV #energy_ranges
    i1 = fin['i1'][:]*mm #y_displacement_ranges
    edep = fin['compton-electron/edep'][:]*MeV
    ybin = fin['compton-electron/ybin'][:]*mm
    zbin = fin['compton-electron/zbin'][:]*mm
    # sum over scintillator thickness
    edep = edep.sum(axis=2)
    # In our internal units, R matrix (joules in bin per incident gamma)
    # has very small numbers, so we should use 64-bit floats.
    edep = edep.astype('float64')
    assert(edep.size % num_gamma_bins == 0)
    num_scint_bins = edep.size // num_gamma_bins
    R = np.reshape(edep, (num_gamma_bins, num_scint_bins)).copy()
    R = R/num_events[:,np.newaxis]
    return R, Bins((i0, i1)), Bins((ybin, zbin))

def iterate_shepp_vardi(x0, R, y):
    y0 = np.matmul(x0, R)
    mask = (y0 != 0)
    yrat = np.zeros_like(y)
    yrat[mask] = y[mask]/y0[mask]
    return (x0/R.sum(axis=1)) * np.matmul(R, yrat)

def main():
    def photon_spectral_density(E):
        E0 = 2*MeV
        sigma0 = 1*MeV
        total_num_photons = 1e10
        A0 = total_num_photons/np.sqrt(twopi*sigma0**2)
        return A0 * np.exp(-(E-E0)**2/(2*sigma0**2))

    # with h5py.File('response-4500A.h5', 'r') as fin:
    with h5py.File('../../deposition/4500-v3/out/4500.h5', 'r') as fin:
        R, gamma_bins, scint_bins = load_response(fin)

    x0 = np.zeros(gamma_bins.num_index)
    dE = []
    for i in gamma_bins:
        energy, ydisp = gamma_bins.index2value(i)
        binwidth = gamma_bins.index2binwidth(i)
        dE.append(binwidth[0])
        x0[i] = photon_spectral_density(energy)*binwidth[0]
    y0 = np.matmul(x0, R)
    dE = np.array(dE)

    # y_out = (y0.reshape((300, 450))).sum(axis=0)
    # np.savetxt('y.dat', y_out/GeV)

    x_experiment = np.zeros(gamma_bins.num_index)
    E0 = 1.8*MeV
    E1 = 2*MeV
    print(gamma_bins.value2index((E0, 0*mm)))
    print(gamma_bins.value2index((E1, 0*mm)))
    x_experiment[gamma_bins.value2index((E0, 0*mm))] = 1e10
    x_experiment[gamma_bins.value2index((E1, 0*mm))] = 1e10
    y_experiment = np.matmul(x_experiment, R)
    y_out = (y_experiment.reshape((300, 450))).sum(axis=0)
    np.savetxt('y_experiment.dat', y_out/GeV)

    with h5py.File('pwfa.h5', 'r') as fin:
        y_experiment = fin['/compton-electron/edep'][0,:]*MeV
        y_experiment = y_experiment.sum(axis=0).flatten()
        y_experiment = y_experiment.astype('float64')

    x = x0
    for i in range(50):
        np.savetxt('{}.dat'.format(i), x)
        x = iterate_shepp_vardi(x, R, y_experiment)
        binz = np.arange(len(x))
        x *= np.tanh(binz/30)
    photon_spectral_density = x/dE

    with h5py.File('spectrum.h5', 'w') as fout:
        fout['energy_bins'] = gamma_bins.bins[0]/MeV
        fout['photon_spectral_density'] = photon_spectral_density/(1/MeV)


if __name__ == '__main__':
    sys.exit(main())
