# -*- coding: utf-8 -*-
import os, sys, random
import argparse
import numpy as np
import toml
import asteval
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
    ax.plot(zscint/mm, yscint/mm, color='#cccccc', linewidth=0.4, zorder=1)
    ax.plot(zscint/mm, -yscint/mm, color='#cccccc', linewidth=0.4, zorder=1)

def plot_deposition_2d(filename, conf, edep, num_events, xbin, ybin, zbin):

    mpl.rc('figure.subplot', right=0.99, top=0.85, bottom=0.14, left=0.10)
    fig = plot.figure(figsize=(244.0/72, 163.0/72))
    ax = fig.add_subplot(1, 1, 1) #, aspect=1)

    edep = edep.sum(axis=conf['Files']['ProjectionAxis'])
    Nbin = np.array((1, 1))
    assert((np.array(edep.shape) % Nbin).sum() == 0)
    edep = edep.reshape(
        edep.shape[0]//Nbin[0], Nbin[0], edep.shape[1]//Nbin[1], Nbin[1])
    edep = edep.sum(axis=3).sum(axis=1)
    # edep /= num_events

    if conf['Files']['TransposeProjection'] == True:
        edep = edep.T

    if conf['Colorbar']['Normalize']:
        vmax = 1.0
        plot_val = edep.T/edep.max()
        cb_label = 'Energy deposited (arb)'
    else:
        gammas_per_shot = conf['Files']['GammasPerShot']
        vmax = edep.max()/(GeV/gammas_per_shot)
        plot_val = edep.T/(GeV/gammas_per_shot)
        cb_label = 'GeV deposited per ' + conf['Annotation']['ShotLabel']
    image = ax.imshow(
        plot_val, cmap=common.blue_cmap,
        extent=(zbin[0]/mm, zbin[-1]/mm, ybin[0]/mm, ybin[-1]/mm),
        vmax=vmax, aspect='equal', interpolation='none')  #'auto')
    cb = fig.colorbar(image, shrink=0.95)
    cb.set_label(cb_label, rotation=270, labelpad=10)

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

    # create safe interpreter for evaluation of configuration expressions
    aeval = asteval.Interpreter(use_numpy=True)
    for q in g4.hepunit.__dict__:
        aeval.symtable[q] = g4.hepunit.__dict__[q]
    edep_sum = edep.sum()
    aeval.symtable['edep_sum'] = edep_sum
    edep_avg = (edep/num_events).sum()
    aeval.symtable['edep_avg'] = edep_avg

    if 'Title' in conf['Annotation']:
        plot.title(conf['Annotation']['Title'], fontsize=7)

    if 'Text' in conf['Annotation']:
        text = ''
        for s in conf['Annotation']['Text']:
            text += '\n' + eval(s)
        ax.text(
            *conf['Annotation']['loc'], text, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom')

    if 'EnergyScale' in conf:
        electron_energy_scale_coeff = np.array((
            conf['EnergyScale']['Coefficients']))
        c0, c1, c2, c3 = electron_energy_scale_coeff
        def x_to_electron_energy(x):
            xp = x/mm
            return MeV*np.exp(c0 + c1*xp + c2*xp**2 + c3*xp**3)

        xvals = mm*np.arange(-400, 400, 1)
        gamma_energy_to_x = interp1d(
            compton.edge_to_gamma(
                x_to_electron_energy(xvals)), xvals)
        electron_energy_to_x = interp1d(x_to_electron_energy(xvals), xvals)
        energy_to_x = {
            'Electron' : electron_energy_to_x,
            'Gamma' : gamma_energy_to_x }[conf['EnergyScale']['Type']]

        ax2 = ax.twiny()
        major_vals = np.array((0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 25, 30))*MeV
        minor_vals = np.array(
            (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
             0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9,
             2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19,
             21, 22, 23, 24, 26, 27, 28, 29))*MeV
        ax2.set_xticks(energy_to_x(major_vals)/mm, minor=False)
        ax2.set_xticks(energy_to_x(minor_vals)/mm, minor=True)
        tick_labels = ['{:g}'.format(x) for x in major_vals]
        tick_labels[0] = ''
        ax2.set_xticklabels(tick_labels)

        ax2.set_xlabel('{} Energy (MeV)'.format(conf['EnergyScale']['Type']))

        ax2.set_xlim(ax.get_xlim())
    # print(ax.get_xlim())
    # ax2 = ax.twiny()
    # ax2.set_xlim(0.0, 225.0)
    plot.savefig(filename, transparent=True) #, dpi=100)
    plot.close()
#    output.savefig(fig, transparent=False)

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
#    output = PdfPages(filename)
    plot_deposition_2d(filename, conf, edep, num_events, xbin, ybin, zbin)
#    output.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())
