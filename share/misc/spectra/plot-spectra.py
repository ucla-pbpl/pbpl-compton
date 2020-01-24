#!/usr/bin/env python
import sys
import os
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import simps
from scipy.interpolate import griddata
from pbpl import common
from pbpl.common.units import *

mrad = 1e-3
steradian = 1.0

def log_minor_ticks(axis):
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
    axis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
    axis.set_minor_locator(locmin)
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())

def plot_spot(
        output, fin, group_name, num_contours, zscale, xlim, ylim, label=None):
    g = fin[group_name]
    photon_energy = g['energy'][:]*MeV
    thetax = g['thetax'][:]*mrad
    thetay = g['thetay'][:]*mrad
    d2W = g['d2W'][:]*joule/(mrad**2*MeV)
    dthetax = thetax[1]-thetax[0]
    dthetay = thetay[1]-thetay[0]

    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel(r'$\theta_x$ (mrad)', labelpad=0.0)
    plot.ylabel(r'$\theta_y$ (mrad)', labelpad=0.0)

    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    post_shape = np.array(list(itertools.product(x, y)))

    foo = np.array(list(itertools.product(thetax/mrad, thetay/mrad)))
    bar = simps(d2W, photon_energy, axis=0).flatten()

    A = griddata(foo, bar, post_shape, method='cubic')
    mask = (A < A.max()*0.001)
    A[mask] = A.max()*0.001

    contours = ax.contourf(
        x, y, A.reshape((100,100)).T / zscale[0],
        levels=num_contours, cmap=common.blue_cmap, vmin=0)
    ax.contour(
        x, y, A.reshape((100,100)).T / zscale[0],
        levels=num_contours, colors='k', linewidths=0.4, vmin=0)

    cbar = plot.colorbar(contours)
    cbar.ax.set_ylabel(zscale[1])
    if label == None:
        label = group_name
    ax.text(0.02, 0.9, label, transform=ax.transAxes, size=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    output.savefig(fig, transparent=True)

def plot_double_differential(
        output, fin, group_name, num_contours,
        xscale, yscale, zscale, xlim, ylim, label=None):
    g = fin[group_name]
    photon_energy = g['energy'][:]*MeV
    thetax = g['thetax'][:]*mrad
    thetay = g['thetay'][:]*mrad
    d2W = g['d2W'][:]*joule/(mrad**2*MeV)
    dthetax = thetax[1]-thetax[0]
    dthetay = thetay[1]-thetay[0]

    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel(r'Photon energy ({})'.format(xscale[1]), labelpad=0.0)
    plot.ylabel(r'$\theta_x$ ({})'.format(yscale[1]), labelpad=0.0)

    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    post_shape = np.array(list(itertools.product(x, y)))

    foo = np.array(list(itertools.product(
        photon_energy/xscale[0], thetax/yscale[0])))
    bar = simps(d2W, thetay, axis=2).flatten()
    A = griddata(foo, bar, post_shape, method='cubic')
    mask = (A < A.max()*0.001)
    A[mask] = A.max()*0.001

    contours = ax.contourf(
        x, y, A.reshape((100,100)).T / zscale[0],
        levels=num_contours, cmap=common.blue_cmap, vmin=0)
    ax.contour(
        x, y, A.reshape((100,100)).T / zscale[0],
        levels=num_contours, colors='k', linewidths=0.4, vmin=0)

    cbar = plot.colorbar(contours)
    cbar.ax.set_ylabel(zscale[1])

    if label == None:
        label = group_name
    ax.text(0.02, 0.9, label, transform=ax.transAxes, size=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    output.savefig(fig, transparent=True)


def plot_spectral_energy_density(
        output, fin, group_names, xscale, yscale, xlim, ylim,
        labels=None, display_legend=True):
    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel(r'Photon energy ({})'.format(xscale[1]), labelpad=0.0)
    plot.ylabel(
        r'Energy spectral density ({})'.format(yscale[1]), labelpad=1.0)

    for i, group_name in enumerate(group_names):
        g = fin[group_name]
        photon_energy = g['energy'][:]*MeV
        thetax = g['thetax'][:]*mrad
        thetay = g['thetay'][:]*mrad
        d2W = g['d2W'][:]*joule/(mrad**2*MeV)
        dthetax = thetax[1]-thetax[0]
        dthetay = thetay[1]-thetay[0]

        spectral_energy_density = d2W.sum(axis=(1,2))*dthetax*dthetay

        label = group_name
        if labels != None:
            label = labels[i]
        mask = spectral_energy_density>0
        ax.plot(
            np.concatenate(((0.0,), photon_energy[mask]/xscale[0])),
            np.concatenate(((0.0,), spectral_energy_density[mask]/yscale[0])),
            linewidth=0.6,
            label=label)

    if display_legend:
        ax.legend()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    output.savefig(fig, transparent=True)

def plot_spectral_photon_density(
        output, fin, group_names, xscale, yscale, xlim, ylim,
        labels=None, display_legend=True):
    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel(r'Photon energy ({})'.format(xscale[1]), labelpad=0.0)
    plot.ylabel(
        r'Spectral photon density ({})'.format(yscale[1]), labelpad=1.0)

    for i, group_name in enumerate(group_names):
        g = fin[group_name]
        photon_energy = g['energy'][:]*MeV
        thetax = g['thetax'][:]*mrad
        thetay = g['thetay'][:]*mrad
        d2W = g['d2W'][:]*joule/(mrad**2*MeV)
        dthetax = thetax[1]-thetax[0]
        dthetay = thetay[1]-thetay[0]

        spectral_energy_density = d2W.sum(axis=(1,2))*dthetax*dthetay
        spectral_photon_density = spectral_energy_density/photon_energy

        label = group_name
        if labels != None:
            label = labels[i]

        mask = spectral_photon_density>0
        ax.semilogy(
            np.concatenate(((0.0,), photon_energy[mask]/xscale[0])),
            np.concatenate(((0.0,), spectral_photon_density[mask]/yscale[0])),
            linewidth=0.6,
            label=label)

        num_photons = simps(spectral_photon_density, photon_energy)

    if display_legend:
        ax.legend()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    log_minor_ticks(ax.yaxis)

    output.savefig(fig, transparent=True)


def main():
    fin = h5py.File('d2W.h5', 'r')
    common.setup_plot()
    os.makedirs('figs', exist_ok=True)

    # PWFA
    if True:
        pwfa_drive_datasets = [
            'E300 PWFA/Unmatched Trailing (s=0.17)/drive',
            'E300 PWFA/Unmatched Trailing (s=0.14)/drive',
            'E300 PWFA/Matched Trailing (s=0.06)/drive',
            'E300 PWFA/Unmatched Trailing (s=0.01)/drive']
        pwfa_trailing_datasets = [
            'E300 PWFA/Unmatched Trailing (s=0.17)/trailing',
            'E300 PWFA/Unmatched Trailing (s=0.14)/trailing',
            'E300 PWFA/Matched Trailing (s=0.06)/trailing',
            'E300 PWFA/Unmatched Trailing (s=0.01)/trailing']
        output = PdfPages('figs/pwfa.pdf')
        plot_spectral_energy_density(
            output, fin, pwfa_drive_datasets,
            (MeV, 'MeV'),
            (mJ/MeV, 'mJ/MeV'),
            (-0.1, 2.0), (-0.05, 2.6))
        plot_spectral_energy_density(
            output, fin, pwfa_trailing_datasets,
            (MeV, 'MeV'),
            (mJ/MeV, 'mJ/MeV'),
            (-0.1, 2.0), (-0.05, 2.6))
        plot_spectral_photon_density(
            output, fin, pwfa_drive_datasets,
            (MeV, 'MeV'),
            (1/MeV, '1/MeV'),
            (-0.1, 2.0), (1e7, 1e13))
        plot_spectral_photon_density(
            output, fin, pwfa_trailing_datasets,
            (MeV, 'MeV'),
            (1/MeV, '1/MeV'),
            (-0.1, 2.0), (1e7, 1e13))
        for x in pwfa_drive_datasets:
            plot_spot(
                output, fin, x, 6,
                (mJ/mrad**2, r'mJ/mrad$^2$'),
                (-2.0, 2.0), (-1.5, 1.5))
        for x in pwfa_trailing_datasets:
            plot_spot(
                output, fin, x, 6,
                (mJ/mrad**2, r'mJ/mrad$^2$'),
                (-2.0, 2.0), (-1.5, 1.5))
        for x in pwfa_drive_datasets:
            plot_double_differential(
                output, fin, x, 8,
                (MeV, 'MeV'), (mrad, 'mrad'),
                (uJ/(mrad*MeV), r'uJ/(mrad$^\cdot$MeV)'),
                (0.01, 1.0), (-2.0, 2.0))
        for x in pwfa_trailing_datasets:
            plot_double_differential(
                output, fin, x, 8,
                (MeV, 'MeV'), (mrad, 'mrad'),
                (uJ/(mrad*MeV), r'uJ/(mrad$^\cdot$MeV)'),
                (0.01, 1.0), (-2.0, 2.0))
        output.close()

    # Filamentation
    if True:
        output = PdfPages('figs/filamentation.pdf')
        plot_spectral_energy_density(
            output, fin,
            ['Filamentation/solid'],
            (MeV, 'MeV'), (uJ/MeV, 'uJ/MeV'),
            (-4, 60.0), (-5, 100), display_legend=False)
        plot_spectral_photon_density(
            output, fin,
            ['Filamentation/solid'],
            (MeV, 'MeV'),
            (1/MeV, '1/MeV'),
            (-2, 60.0), (1e6, 1e10), display_legend=False)
        plot_spot(
            output, fin, 'Filamentation/solid', 5,
            (mJ/mrad**2, r'mJ/mrad$^2$'),
            (-0.2, 0.2), (-0.2, 0.2), '')
        plot_double_differential(
            output, fin, 'Filamentation/solid', 8,
            (MeV, 'MeV'), (mrad, 'mrad'),
            (uJ/(mrad*MeV), r'uJ/(mrad$\cdot$MeV)'),
            (1, 30), (-0.2, 0.2), '')
        output.close()

    # SFQED
    if True:
        output = PdfPages('figs/sfqed.pdf')
        sfqed_datasets = [
            'SFQED/MPIK/LCFA_w3.0_xi5.7',
            'SFQED/MPIK/LCFA_w2.4_xi7.2',
            'SFQED/MPIK/LCS+LCFA_w3.0_xi5.7',
            'SFQED/MPIK/LCS+LCFA_w2.4_xi7.2']
        sfqed_labels = [
            r'LCFA ($a_0=5.7, w_0=3.0\;\mu{\rm m}$)',
            r'LCFA ($a_0=7.2, w_0=2.4\;\mu{\rm m}$)',
            r'QED ($a_0=5.7, w_0=3.0\;\mu{\rm m}$)',
            r'QED ($a_0=7.2, w_0=2.4\;\mu{\rm m}$)']
        plot_spectral_energy_density(
            output, fin, sfqed_datasets,
            (GeV, 'GeV'), (mJ/GeV, 'mJ/GeV'),
            (-0.1, 10.0), (-0.1, 4),
            labels=sfqed_labels)
        plot_spectral_energy_density(
            output, fin, sfqed_datasets,
            (MeV, 'MeV'), (uJ/MeV, 'uJ/MeV'),
            (-1, 60.0), (-0.1, 3),
            labels=sfqed_labels)
        plot_spectral_photon_density(
            output, fin, sfqed_datasets,
            (GeV, 'GeV'), (1/GeV, '1/GeV'),
            (-0.1, 10.0), (1e1, 1e9),
            labels=sfqed_labels)
        plot_spectral_photon_density(
            output, fin, sfqed_datasets,
            (MeV, 'MeV'), (1/MeV, '1/MeV'),
            (-1, 60.0), (2e4, 4e7),
            labels=sfqed_labels)
        for x, y in zip(sfqed_datasets, sfqed_labels):
            plot_double_differential(
                output, fin, x, 8,
                (MeV, 'MeV'), (mrad, 'mrad'),
                (uJ/(mrad*MeV), r'uJ/(mrad$\cdot$MeV)'),
                (1, 30), (-0.5, 0.5),
                label=y)
        for x, y in zip(sfqed_datasets, sfqed_labels):
            plot_spot(
                output, fin, x, 6,
                (mJ/mrad**2, r'mJ/mrad$^2$'),
                (-0.3, 0.3), (-0.1, 0.1), label=y)
        output.close()

if __name__ == '__main__':
    sys.exit(main())
