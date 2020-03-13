# -*- coding: utf-8 -*-
import os, sys, random
import argparse
import numpy as np
import toml
from pbpl import compton
import Geant4 as g4
from Geant4.hepunit import *
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import pbpl.common as common
from scipy.interpolate import interp1d

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Plot energy deposition density',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-compton-plot-deposition plot-deposition.toml
''')
    parser.add_argument(
        'config_filename', metavar='conf-file',
        help='Configuration file')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args.conf = toml.load(args.config_filename)
    return args

def plot_quadrupole_magnet_profile(ax, conf):
    mag_d0 = conf['d0'] * mm
    mag_c1 = conf['c1'] * mm
    profile_zp = 0.5 * mag_d0 / (z/mag_c1)
    zp = np.linspace(0, 320, 320) * mm
    z = 10.0 + np.cos(40.0*deg) * zp
    ax.plot(zp/mm, profile_zp/mm, color='k', linewidth=0.4)
    ax.plot(zp/mm, -profile_zp/mm, color='k', linewidth=0.4)

def plot_sextupole_magnet_profile(ax, conf):
    from numpy.linalg import inv
    M = compton.build_transformation(conf['Transformation'], mm, deg)
    Minv = inv(M)
    c0 = conf['c0'] * mm
    b1 = conf['b1'] * mm
    xoff = conf['xoff'] * mm
    alpha0 = conf['alpha0'] * deg
    a0 = c0 * ( 3 * (b1/c0)**2 - 1)**(1.0/3)
    y = np.arange(1*mm, 200*mm, 1*mm)
    z = np.sqrt( (y**3 + a0**3)/(3*y) )
    x = xoff + np.tan(alpha0)*z
    pos = np.array((x, y, z))
    tpos = np.array([compton.transform(M, x) for x in pos.T]).T
    _, yscint, zscint = tpos
    ax.plot(zscint/mm, yscint/mm, color='k', linewidth=0.4)
    ax.plot(zscint/mm, -yscint/mm, color='k', linewidth=0.4)

def plot_deposition_2d(output, conf, edep, num_events, xbin, ybin, zbin):

    mpl.rc('figure.subplot', right=0.99, top=0.85, bottom=0.14, left=0.10)
    fig = plot.figure(figsize=(244.0/72, 150.0/72))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    # plot.subplots_adjust(top=0.85, bottom=0.14)
#    plot.subplots_adjust(bottom=0.14)

    gammas_per_shot = conf['Files']['GammasPerShot']
    edep = edep.sum(axis=conf['Files']['ProjectionAxis'])
    edep /= num_events
    if conf['Files']['TransposeProjection'] == True:
        edep = edep.T

    image = ax.imshow(
        edep.T/(GeV/gammas_per_shot), cmap=common.blue_cmap,
        extent=(zbin[0]/mm, zbin[-1]/mm, ybin[0]/mm, ybin[-1]/mm),
        vmax=edep.max()/(GeV/gammas_per_shot), aspect='auto')

    cb = fig.colorbar(image, shrink=0.95)
    cb.set_label(
        'GeV deposited per ' + conf['Annotation']['ShotLabel'],
        rotation=270, labelpad=10)

    if 'QuadrupoleMagnetProfile' in conf:
        plot_quadrupole_magnet_profile(ax, conf['QuadrupoleMagnetProfile'])

    if 'SextupoleMagnetProfile' in conf:
        plot_sextupole_magnet_profile(ax, conf['SextupoleMagnetProfile'])


    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.set_xlabel(r"$z_{\rm scint}$ (mm)", labelpad=0.0)
    ax.set_ylabel(r'$y_{\rm scint}$ (mm)', labelpad=0.0)

    ax.set_xlim(zbin[0]/mm, zbin[-1]/mm)
    ax.set_ylim(ybin[0]/mm, ybin[-1]/mm)

    if 'Title' in conf['Annotation']:
        plot.title(conf['Annotation']['Title'], fontsize=7)
    if 'Text' in conf['Annotation']:
        text = ''
        for s in conf['Annotation']['Text']:
            text += '\n' + s
        ax.text(*conf['Annotation']['loc'], text, transform=ax.transAxes)

    if 'EnergyScale' in conf:

        electron_energy_scale_coeff = np.array((
            -3.773365167285961,
            0.04696056580188213,
            -0.0001345372207283992,
            1.774027873917722e-7))

        c0, c1, c2, c3 = electron_energy_scale_coeff

        def gamma_energy_to_electron_energy(gamma_energy):
            return gamma_energy*(
                1-1/(1 + 2*gamma_energy/electon_mass_c2))

        def electron_energy_to_gamma_energy(electron_energy):
            return 0.5*(electron_energy + np.sqrt(
                electron_energy**2 + 2 * electron_energy * electron_mass_c2))

        def x_to_electron_energy(x):
            xp = x/mm
            return MeV*np.exp(c0 + c1*xp + c2*xp**2 + c3*xp**3)

        xvals = mm*np.linspace(1, 301, 300)
        gamma_energy_to_x = interp1d(
            electron_energy_to_gamma_energy(
                x_to_electron_energy(xvals)), xvals)
        electron_energy_to_x = interp1d(x_to_electron_energy(xvals), xvals)

        energy_to_x = gamma_energy_to_x

        ax2 = ax.twiny()
        # energy_vals = MeV*np.array(
        #     (0.25, 0.5, 1, 2, 4.0), dtype=float)
        energy_vals = MeV*np.array(
            (0.1, 0.2, 0.4,0.8, 1.6, 3.2, 6.4), dtype=float)
        def subdivide(x, N):
            if N>1:
                y = subdivide(x, N-1)
                return np.sort(np.concatenate((y, 0.5*(y[1:] + y[:-1]))))
            else:
                return x
        ax2.set_xticks(energy_to_x(energy_vals)/mm, minor=False)
        ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax2.set_xticks(
            energy_to_x(np.array(subdivide(energy_vals, 3)))/mm, minor=True)
        ax2.set_xticklabels([str(x) for x in energy_vals])
        ax2.set_xlabel('Gamma Energy (MeV)')
        ax2.set_xlim(ax.get_xlim())

    output.savefig(fig, transparent=False)

def main():
    args = get_args()
    conf = args.conf
    fin = h5py.File(conf['Files']['Input'], 'r')
    run_index = tuple(conf['Files']['RunIndex'])
    num_events = fin['num_events'][run_index]
    gin = fin[conf['Files']['Group']]
    edep = gin['edep'][run_index]*MeV
    xbin = gin['xbin'][:]*mm
    ybin = gin['ybin'][:]*mm
    zbin = gin['zbin'][:]*mm

    common.setup_plot()

    filename = conf['Files']['Output']
    path = os.path.dirname(filename)
    if path != '':
        os.makedirs(path, exist_ok=True)
    output = PdfPages(filename)
    plot_deposition_2d(output, conf, edep, num_events, xbin, ybin, zbin)
    output.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())
