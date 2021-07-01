#!/usr/bin/env python
import sys
import toml
import numpy as np
from scipy.interpolate import interp1d
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
from pbpl import common
from pbpl.common.units import *
from scipy.ndimage import convolve1d
from scipy.signal import gaussian

def dump(output, conf, page_conf):
    i0 = page_conf['i0']
    with h5py.File(conf['Input'], 'r') as fin:
        edep = fin['edep'][i0,:]*MeV
        theta0_vals = fin['i1'][:]*deg
        xbin = fin['xbin'][:]*mm
        ybin = fin['ybin'][:]*mm
        zbin = fin['zbin'][:]*mm
    dz = zbin[1] - zbin[0]

    with h5py.File(conf['Normalization'], 'r') as fin:
        edep_norm = fin['edep'][i0,:]*MeV
        assert(edep_norm.shape == edep.shape)

    edep = edep.sum(axis=(-3,-2))
    edep_norm = edep_norm.sum(axis=(-3,-2))
    mask = (edep_norm != 0.0)
    normalized_edep = np.zeros_like(edep)
    normalized_edep[mask] = edep[mask]/edep_norm[mask]

    sigma = conf['SmoothingSigma']*mm
    sigma_samples = sigma/dz
    window = gaussian(len(zbin), sigma_samples)
    window = window/window.sum()
    smoothed = convolve1d(normalized_edep, window)

    y = 0.5*(ybin[0:-1] + ybin[1:])
    z = 0.5*(zbin[0:-1] + zbin[1:])

    fig = plot.figure(figsize=(244.0/72, 160.0/72))
    plot.subplots_adjust(left=0.11, bottom=0.12, top=0.86, right=0.98)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel(r"$z_{\rm scint}$ (mm)", labelpad=-1.0)
    ax.set_ylabel(r'$E_{\rm coll} / E_{\rm uncoll}$', labelpad=1.0)

    for i0, theta0 in enumerate(theta0_vals):
        label = r'$\theta_0 = {}^\circ$'.format(int(round(theta0/deg)))
        ax.plot(z/mm, smoothed[i0], linewidth=0.4, label=label)
    ax.legend(fontsize=7, labelspacing=0.0)
    ax.set_xlim(*conf['xlim'])
    ax.set_ylim(*conf['ylim'])

    text = ''
    for s in page_conf['Annotation']['Text']:
        text += '\n' + s
    ax.text(
        *page_conf['Annotation']['Location'], text, fontsize=7,
        verticalalignment='top', transform=ax.transAxes)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    energy_scale_coeff = np.array(page_conf['EnergyScaleCoeff'])
    c0, c1, c2, c3 = energy_scale_coeff
    def z_to_energy(z):
        z0 = z/mm
        return np.exp(c0 + c1*z0 + c2*z0**2 + c3*z0**3)*MeV

    zvals = np.arange(-1000, 1000, 1)*mm
    energy_to_z = interp1d(z_to_energy(zvals), zvals)

    ax2 = ax.twiny()
    major_energy = np.array((0.1, 0.5, 1.0, 5, 10.0, 15.0, 20.0, 25.0))*MeV
    ax2.set_xticks(energy_to_z(major_energy)/mm, minor=False)
    ticklabels = ['{:g}'.format(x/MeV) for x in major_energy]
    ticklabels = [x if len(x)<7 else '' for x in ticklabels]
    ax2.set_xticklabels(ticklabels)
    minor_energy = np.array(
        (0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 2, 3, 4, 6, 7, 8, 9,
         11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29))*MeV
    ax2.set_xticks(energy_to_z(minor_energy)/mm, minor=True)
    ax2.set_xlabel('Electron energy (MeV)')
    ax2.set_xlim(ax.get_xlim())

    output.savefig(fig, transparent=True)

def main():
    conf = toml.load(sys.argv[1])

    common.setup_plot()
    with PdfPages(conf['Output']) as output:
        for page_conf in conf['Pages']:
            dump(output, conf, page_conf)


if __name__ == '__main__':
    sys.exit(main())
