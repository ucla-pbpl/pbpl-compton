#!/usr/bin/env python
import sys
import numpy as np
from scipy.interpolate import interp1d
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
from pbpl import common
from pbpl.common.units import *

def dump(
        output, collimated_filename, uncollimated_filename,
        title):
    with h5py.File(collimated_filename, 'r') as fin:
        edep = fin['edep'][:]*eV
        theta0_vals = fin['i0'][:]*deg
        xbin = fin['xbin'][:]*mm
        ybin = fin['ybin'][:]*mm
        zbin = fin['zbin'][:]*mm

    with h5py.File(uncollimated_filename, 'r') as fin:
        uncollimated_edep = fin['edep'][:]*eV
        assert(edep.shape == uncollimated_edep.shape)

    edep = edep.sum(axis=(-3,-2))
    uncollimated_edep = uncollimated_edep.sum(axis=(-3,-2))
    normalized_edep = edep/uncollimated_edep
    # import ipdb
    # ipdb.set_trace()
    y = 0.5*(ybin[0:-1] + ybin[1:])
    z = 0.5*(zbin[0:-1] + zbin[1:])

    fig = plot.figure(figsize=(244.0/72, 160.0/72))
    plot.subplots_adjust(top=0.8)
    ax = fig.add_subplot(1, 1, 1) #, position=(0.2,0.20.97))

    ax.set_xlabel(r"$z/\cos \,\theta'$ (mm)", labelpad=-1.0)
    ax.set_ylabel(r'$E_{\rm coll} / E_{\rm uncoll}$', labelpad=1.0)

    for i0, theta0 in enumerate(theta0_vals):
        if i0 in [0, 1, 5]:
            label = r'$\theta_0 = {}^\circ$'.format(int(round(theta0/deg)))
        else:
            label = None
        ax.plot(z/mm, normalized_edep[i0], linewidth=0.4, label=label)
    ax.legend(fontsize=7)
    # ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.0)

    ax.text(
        0.3, 0.93, title, fontsize=7,
        verticalalignment='top', transform=ax.transAxes)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    # ax2 = ax.secondary_xaxis(
    #     'top', functions=(
    #         lambda x : np.exp(-1.53 + 0.0309*x - 5.08e-5*x**2),
    #         lambda x : x))
    def x_to_energy(x):
        return np.exp(-1.53 + 0.0309*x - 5.08e-5*x**2)
    xvals = np.linspace(1, 301, 300)
    energy_to_x = interp1d(x_to_energy(xvals), xvals)

    ax2 = ax.twiny()
    energy_vals = np.array((0.25, 0.5, 1, 2, 4, 8, 16), dtype=float)
    def subdivide(x, N):
        if N>1:
            y = subdivide(x, N-1)
            return np.sort(np.concatenate((y, 0.5*(y[1:] + y[:-1]))))
        else:
            return x
    ax2.set_xticks(energy_to_x(energy_vals), False)
#    ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax2.set_xticks(
        energy_to_x(np.array(subdivide(energy_vals, 3))), minor=True)
    ax2.set_xticklabels([str(x) for x in energy_vals])
    ax2.set_xlabel('Energy (MeV)')
    ax2.set_xlim(ax.get_xlim())

    output.savefig(fig, transparent=True)

def main():
    common.setup_plot()

    with PdfPages('collimation-output.pdf') as output:
        dump(output, 'results/W-5-p45-10.h5', 'results/uncollimated.h5',
             'Tungsten collimator\n' +
             r'$a = 5\,{\rm mm}, r = 0.45\,a, d = 10\,{\rm mm}$')
        dump(output, 'results/W-4-p43-12.h5', 'results/uncollimated.h5',
             'Tungsten collimator\n' +
             r'$a = 4\,{\rm mm}, r = 0.43\,a, d = 12\,{\rm mm}$')
        dump(output, 'results/W-5-p42-15.h5', 'results/uncollimated.h5',
             'Tungsten collimator\n' +
             r'$a = 5\,{\rm mm}, r = 0.42\,a, d = 15\,{\rm mm}$')
        dump(output, 'results/Al-5-p45-10.h5', 'results/uncollimated.h5',
             'Aluminum collimator\n' +
             r'$a = 5\,{\rm mm}, r = 0.45\,a, d = 10\,{\rm mm}$')



if __name__ == '__main__':
    sys.exit(main())
