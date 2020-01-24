#!/usr/bin/env python
import sys
import tqdm
import time
import itertools
import numpy as np
import h5py

MeV = 1.0
mrad = 1.0
joule = 1.0
eV = 1e-6*MeV
rad = mrad*1e3
steradian = rad*rad

def create_dset(f, name, A):
    dset = f.create_dataset(
        name, shape=A.shape, dtype=A.dtype, compression='gzip')
    dset[:] = A

def dump_dsets(fout, dir_name, subdir_name, energy, thetax, thetay, d2W):
    create_dset(
        fout, '{}/{}/energy'.format(dir_name, subdir_name), energy/MeV)
    create_dset(
        fout, '{}/{}/thetax'.format(dir_name, subdir_name), thetax/mrad)
    create_dset(
        fout, '{}/{}/thetay'.format(dir_name, subdir_name), thetay/mrad)
    create_dset(
        fout, '{}/{}/d2W'.format(dir_name, subdir_name),
        d2W/(joule/(mrad**2*MeV)))

def main():
    fout = h5py.File('d2W.h5', 'w')
    fmt = ('{desc:40s} {percentage:3.0f}% ' +
           '|{bar}| {n_fmt:>4s}/{total_fmt:<4s}')
    data_dir = 'raw'

    def conv_pwfa(energy, thetax, thetay, d2W):
        return (energy*MeV, thetax*mrad, thetay*mrad, d2W*joule/(mrad**2*MeV))

    def conv_filamentation(energy, thetax, thetay, d2W):
        return (energy*eV, thetax*mrad, thetay*mrad, d2W*joule/(steradian*eV))

    def conv_mpik(energy, thetax, thetay, d2W):
        return (np.power(10, energy)*MeV,
                thetax*mrad, thetay*mrad, d2W*joule/(mrad**2*MeV))

    def conv_ist(energy, thetax, thetay, d2W):
        return (energy*MeV, thetax*mrad, thetay*mrad, d2W*joule/(mrad**2*MeV))

    PWFA = list(itertools.product(
        ['E300 PWFA/Matched Trailing (s=0.06)',
         'E300 PWFA/Unmatched Trailing (s=0.01)',
         'E300 PWFA/Unmatched Trailing (s=0.14)',
         'E300 PWFA/Unmatched Trailing (s=0.17)'],
        ['trailing', 'drive', 'both'],
        [conv_pwfa],
        ['{}/{}/d2W_{}.txt'],
        [',']))

    Filamentation = list(itertools.product(
        ['Filamentation'],
        ['solid'],
        [conv_filamentation],
        ['{}/{}/filamentation_{}_d2W.txt'],
        [',']))

    SFQED_MPIK = list(itertools.product(
        ['SFQED/MPIK'],
        ['LCFA_w2.4_xi7.2',
         'LCFA_w3.0_xi5.7',
         'LCS+LCFA_w2.4_xi7.2',
         'LCS+LCFA_w3.0_xi5.7'],
        [conv_mpik],
        ['{}/{}/d2J_dMeVd2mrad_{}.txt'],
        [None]))

    pbar = tqdm.tqdm(PWFA + Filamentation + SFQED_MPIK, bar_format=fmt)
    for dir_name, subdir_name, converter, filename_format, delimiter in pbar:
        pbar.set_description('{}/{}'.format(dir_name, subdir_name))
        # DropBox data files are written in 'Z-major order' (i.e., C order):
        #  Z is faster index and X is slowest index
        raw = np.loadtxt(
            filename_format.format(
                data_dir, dir_name, subdir_name), delimiter=delimiter,
            dtype=np.float32).T
        energy, thetax, thetay = [np.sort(np.unique(x)) for x in raw[0:3]]
        d2W = raw[3].reshape(len(energy), len(thetax), len(thetay), order='C')
        energy, thetax, thetay, d2W = converter(energy, thetax, thetay, d2W)
        dump_dsets(fout, dir_name, subdir_name, energy, thetax, thetay, d2W)
    pbar.close()

    pbar = tqdm.tqdm(['a0_5', 'a0_8'], bar_format=fmt)
    dir_name = 'SFQED/IST'
    for subdir_name in pbar:
        pbar.set_description('{}/{}'.format(dir_name, subdir_name))

        filename_format = '{}/{}/{}/Radiated_Energy_Cross_Section_low.txt'
        raw_lo = np.loadtxt(
            filename_format.format(data_dir, dir_name, subdir_name),
            skiprows=1, dtype=np.float32).T
        energy_lo, thetax_lo, thetay_lo= [
            np.sort(np.unique(x)) for x in raw_lo[0:3]]

        filename_format = '{}/{}/{}/Radiated_Energy_Cross_Section.txt'
        raw_hi = np.loadtxt(
            filename_format.format(data_dir, dir_name, subdir_name),
            skiprows=1, dtype=np.float32).T
        energy_hi, thetax_hi, thetay_hi= [
            np.sort(np.unique(x)) for x in raw_hi[0:3]]

        assert(np.array_equal(thetax_lo, thetax_hi))
        assert(np.array_equal(thetay_lo, thetay_hi))

        thetax = thetax_lo
        thetay = thetay_lo
        energy = np.concatenate((energy_lo, energy_hi))

        # weird non-C/non-FORTRAN order:  X, then Z, then Y
        d2W_lo = raw_lo[3].reshape(
            len(energy_lo), len(thetax)*len(thetay), order='F')
        d2W_lo = d2W_lo.reshape(
            len(energy_lo), len(thetax), len(thetay), order='C')

        d2W_hi = raw_hi[3].reshape(
            len(energy_hi), len(thetax)*len(thetay), order='F')
        d2W_hi = d2W_hi.reshape(
            len(energy_hi), len(thetax), len(thetay), order='C')

        d2W = np.vstack((d2W_lo, d2W_hi))

        energy, thetax, thetay, d2W = conv_ist(energy, thetax, thetay, d2W)

        dump_dsets(fout, dir_name, subdir_name, energy, thetax, thetay, d2W)
    pbar.close()

    fout.close()


if __name__ == '__main__':
    sys.exit(main())
