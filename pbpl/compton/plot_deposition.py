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
    a0 = c0 * ( 3 * (b1/c0)**2 - 1)**(1.0/3)

    from scipy.interpolate import interp1d
    y = np.arange(1, 90*mm, 1.0*mm)
    z = np.sqrt( (y**3 + a0**3)/(3*y) )
    y_z = interp1d(z, y)
    zscint = np.arange(95, 300, 1.0*mm)
    pos = np.array((np.zeros_like(zscint), np.zeros_like(zscint), zscint))
    tpos = np.array([compton.transform(M, x) for x in pos.T]).T
    tpos[1] = y_z(tpos[2])
    pos = np.array([compton.transform(Minv, x) for x in tpos.T]).T

    ax.plot(pos[2]/mm, pos[1]/mm, color='k', linewidth=0.4)
    ax.plot(pos[2]/mm, -pos[1]/mm, color='k', linewidth=0.4)

def plot_deposition_2d(output, conf, edep, xbin, ybin, zbin):
    mpl.rc('figure.subplot', right=0.99, top=0.97, bottom=0.09, left=0.10)
    fig = plot.figure(figsize=(244.0/72, 150.0/72))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    gammas_per_shot = conf['Files']['GammasPerShot']
    edep = edep.sum(axis=conf['Files']['ProjectionAxis'])
    if conf['Files']['TransposeProjection'] == True:
        edep = edep.T
    # import ipdb
    # ipdb.set_trace()
    image = ax.imshow(
        edep.T/(GeV/gammas_per_shot), cmap=common.blue_cmap,
        extent=(zbin[0]/mm, zbin[-1]/mm, ybin[0]/mm, ybin[-1]/mm),
        vmax=edep[30:].max()/(GeV/gammas_per_shot))
#        extent=(xbin[0]/mm, xbin[-1]/mm, ybin[0]/mm, ybin[-1]/mm))

    cb = fig.colorbar(image, shrink=0.87)
    cb.set_label(
        'GeV deposited per ' + conf['Annotation']['ShotLabel'],
        rotation=270, labelpad=10)

    if 'QuadrupoleMagnetProfile' in conf:
        plot_quadrupole_magnet_profile(ax, conf['QuadrupoleMagnetProfile'])

    if 'SextupoleMagnetProfile' in conf:
        plot_sextupole_magnet_profile(ax, conf['SextupoleMagnetProfile'])

    ax.set_xlim(zbin[0]/mm, zbin[-1]/mm)
    ax.set_ylim(ybin[0]/mm, ybin[-1]/mm)

    if 'Title' in conf['Annotation']:
        plot.title(conf['Annotation']['Title'], fontsize=7)
    if 'Text' in conf['Annotation']:
        text = ''
        for s in conf['Annotation']['Text']:
            text += '\n' + s
        ax.text(*conf['Annotation']['loc'], text, transform=ax.transAxes)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.set_xlabel(r"$z_{\rm scint}$ (mm)", labelpad=0.0)
    ax.set_ylabel(r'$y_{\rm scint}$ (mm)', labelpad=0.0)

    output.savefig(fig, transparent=False)

def main():
    args = get_args()
    conf = args.conf
    fin = h5py.File(conf['Files']['Input'], 'r')
    run_index = tuple(conf['Files']['RunIndex'])
    edep = fin['edep'][run_index]*eV
    xbin = fin['xbin'][:]*mm
    ybin = fin['ybin'][:]*mm
    zbin = fin['zbin'][:]*mm

    common.setup_plot()

    out_fname = conf['Files']['Output']
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    output = PdfPages(out_fname)
    plot_deposition_2d(output, conf, edep, xbin, ybin, zbin)
    output.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())
