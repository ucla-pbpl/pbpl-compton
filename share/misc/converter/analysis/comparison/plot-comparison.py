#!/usr/bin/env python
import sys, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import h5py
from pbpl import common
from Geant4.hepunit import *
import common

def plot_scatter(output, energy, xi, gamma_energy, converter_thickness):
    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel('Energy (MeV)', labelpad=0.0)
    plot.ylabel(r'$\cos \, \theta$', labelpad=0.0)

    ax.plot(
        energy/MeV, xi, marker='o', ls='', fillstyle='full',
        markeredgewidth=0,
        markersize=1.5, alpha=0.4)

    compton = gamma_energy*(1 - (1/(1+(2*gamma_energy/(electron_mass_c2)))))

    ax.text(
        0.03, 0.07,
        'Gamma energy = {:.3f} MeV\n'.format(gamma_energy/MeV) +
        'Compton edge = {:.3f} MeV\n'.format(compton/MeV) +
        'Converter thickness = {} mm'.format(converter_thickness/mm),
        transform=ax.transAxes)
    ax.set_xlim(0, gamma_energy/MeV)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    output.savefig(transparent=True)


def plot_energy_hist(
        output, energy, xi, gamma_energy, converter_thickness,
        num_gammas, theta_max):
    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    normalized_num_gammas = 1e6
    weight_factor = normalized_num_gammas / num_gammas
    plot.xlabel('Energy (MeV)', labelpad=0.0)
    plot.ylabel('Counts per 10$^{6}$ gammas', labelpad=0.0)

    cut_hits = energy[xi>np.cos(theta_max)]
    weights = np.ones_like(cut_hits) * weight_factor
    ax.hist(
        cut_hits/MeV,
        200, (0, gamma_energy/MeV), weights=weights,
        linewidth=0.4, histtype='step')

    ax.text(
        0.03, 0.9,
        r'$\theta \leq ' + '{}'.format(theta_max/deg) + r'^\circ ' +
        r' = \arccos\;' + '{:.3f}'.format(np.cos(theta_max)) + r'$',
        transform=ax.transAxes)

    ax.set_xlim(0, gamma_energy/MeV)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    output.savefig(transparent=True)

def plot_xi_hist(
        output, energy, xi, gamma_energy, converter_thickness,
        num_gammas, theta_max):
    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    normalized_num_gammas = 1e6
    weight_factor = normalized_num_gammas / num_gammas
    plot.xlabel(r'$\cos \, \theta$', labelpad=0.0)
    plot.ylabel('Counts per 10$^{6}$ gammas', labelpad=0.0)

    # cut_hits = energy[xi>np.cos(theta_max)]
    weights = np.ones_like(xi) * weight_factor
    ax.hist(
        xi,
        200, (-1.0, 1.0), weights=weights,
        linewidth=0.4, histtype='step')

    # ax.text(
    #     0.03, 0.9,
    #     r'$\theta \leq ' + '{}'.format(theta_max/deg) + r'^\circ ' +
    #     r' = \arccos\;' + '{:.3f}'.format(np.cos(theta_max)) + r'$',
    #     transform=ax.transAxes)

    # ax.set_xlim(0, gamma_energy/MeV)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    output.savefig(transparent=True)

def plot_run(output, filename, converter_thickness, gamma_energy, particle):
    fin = h5py.File(filename, 'r')
    gin = fin[particle]
    position = gin['position'][:]*mm
    direction = gin['direction'][:]
    energy = gin['energy'][:]*MeV
    num_gammas = fin['num_events'][()]
    fin.close()

    xi = direction.T[2]
    if particle == 'gamma':
        mask = (xi != 1.0)
        position = position[mask]
        direction = direction[mask]
        energy = energy[mask]
        xi = xi[mask]
    # output = PdfPages(os.path.splitext(filename)[0] + '.pdf')
    num_scatter = 2000
    plot_scatter(
        output, energy[:num_scatter], xi[:num_scatter],
        gamma_energy, converter_thickness)
    plot_energy_hist(
        output, energy, xi, gamma_energy, converter_thickness,
        num_gammas, 6*deg)
    plot_xi_hist(
        output, energy, xi, gamma_energy, converter_thickness,
        num_gammas, 6*deg)
    # output.close()

def main():
    common.setup_plot()
    plot.rc('figure.subplot', right=0.96, top=0.97, bottom=0.15, left=0.13)

    output = PdfPages('comparison.pdf')
    for thickness in common.thickness_vals:
        for energy in common.energy_vals:
            for particle in ['e-']:
                desc = '{:.3f}mm_{:.3f}MeV'.format(
                    round(thickness/mm, 3), round(energy/MeV, 3))
                filename = 'out/' + desc + '.h5'
                plot_run(output, filename, thickness, energy, particle)
    output.close()


if __name__ == '__main__':
    sys.exit(main())
