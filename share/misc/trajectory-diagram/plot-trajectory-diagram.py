#!/usr/bin/env python
import sys
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pbpl import common
from pbpl.common.units import *
from scipy.spatial import KDTree
from numpy.linalg import norm
from trajectory import calc_electron_trajectory

def plot_fig(output, dx, E, B, cutoff, energies, alpha0, text, xlim, ylim):
    fig = plot.figure(figsize=(244.0/72, 135.0/72))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    x0 = np.zeros(3)
    for KE in energies:
        for dalpha, linewidth, opacity in zip(
                np.array((-5*deg, 0, 5*deg)),
                [0.2, 0.4, 0.2], [0.5, 1.0, 0.5]):
            alpha = alpha0 + dalpha
            n0 = np.array((-np.sin(alpha), np.cos(alpha), 0))
            traj = calc_electron_trajectory(n0, x0, KE, E, B, dx, cutoff)
            ax.plot(
                traj[0]/mm, traj[1]/mm, linewidth=linewidth,
                color='#0083b8', alpha=opacity)

    alpha = alpha0 - 5*deg
    n0 = np.array((-np.sin(alpha), np.cos(alpha), 0))
    x1 = calc_electron_trajectory(
        n0, x0, energies[-1], E, B, 0.5*mm, cutoff)[0:2].T
    N = len(x1)
    x1 = x1[N//4:N-N//4]

    alpha = alpha0 + 5*deg
    n0 = np.array((-np.sin(alpha), np.cos(alpha), 0))
    x2 = calc_electron_trajectory(
        n0, x0, energies[-1], E, B, 0.5*mm, cutoff)[0:2].T
    N = len(x2)
    x2 = x2[N//4:N-N//4]

    tree = KDTree(x1)
    x2_scan = np.array([tree.query(q) for q in x2])
    i2 = np.argmin(x2_scan.T[0])
    dist, i1 = tree.query(x2[i2])

    x1_f = 1.05*x1[i1]
    line_length = norm(x1_f)
    x1_0 = 0.6*line_length * np.array((-np.sin(alpha0), np.cos(alpha0)))
    ax.plot(
        [0, x1_f[0]/mm], [0, x1_f[1]/mm], color='#888888', linewidth=0.4,
        zorder=-10)
    ax.text(
        x1_f[0]/mm, x1_f[1]/mm, r' $\alpha_f = {:.1f}^\circ$'.format(
            90-np.arctan(x1_f[1]/x1_f[0])/deg), fontsize=7,
        verticalalignment='center')
    ax.plot(
        [0, x1_0[0]/mm], [0, x1_0[1]/mm], color='#888888', linewidth=0.4,
        zorder=-10)
    alpha0_label = r' $\alpha_0 = {:.1f}^\circ$'.format(alpha0/deg)
    ax.text(
        x1_0[0]/mm, x1_0[1]/mm, alpha0_label, fontsize=7,
        verticalalignment='center', horizontalalignment='right')

    plot.xlabel(r'$x$ (mm)', labelpad=0.0)
    plot.ylabel(r'$z$ (mm)', labelpad=0.0)

    ax.text(
        0.03, 0.96, text, fontsize=7,
        verticalalignment='top', transform=ax.transAxes)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    output.savefig(fig, transparent=True)

def plot_quadrupole(output):
    G0 = 3.6*tesla/meter
    dx = 2.5*mm
    def E(x):
        return np.zeros(3)
    def B(X):
        x = X[0]
        y = X[1]
        return np.array((0, 0, -G0*y))
    def cutoff(x, t):
        if x[1] < -10*mm:
            return True
        else:
            return False

    xlim = (-240, 380)
    ylim = (0, 340)
    energies = np.sort((30*MeV) / 2**np.arange(7))

    for alpha0, label in zip(
            np.array((40.7, 30.0, 20.0, 10.0, 0.0, -10.0))*deg,
            ['Enge', None, None, None, 'CPT', None]):
        text = r'$B_y = G_0 z$' + '\n' + r'$G_0 = 3.6\;{\rm T/m}$'
        if label is not None:
            text += '\n' + label
        plot_fig(output, dx, E, B, cutoff, energies, alpha0, text, xlim, ylim)

def plot_sextupole(output):
    G0 = tesla/(235*mm)**2
    dx = 2.5*mm
    def E(x):
        return np.zeros(3)
    def B(X):
        x = X[0]
        y = X[1]
        if y>= 0:
            return np.array((0, 0, -G0*y**2))
        else:
            return np.array((0,0,0))
    def cutoff(x, t):
        if x[1] < -10*mm:
            return True
        else:
            return False

    xlim = (-320, 320)
    ylim = (0, 320)
    energies = np.sort((28*MeV) / 2**np.arange(7))

    for alpha0, label in zip(
            np.array((27.6, 20.0, 10.0, 0.0, -10.0))*deg,
            ['Mirror', None, None, "CPT", None]):
        text = r'$B_y = G_0 z^2$' + '\n' + r'$G_0 = 18.1\;{\rm T/m^2}$'
        if label is not None:
            text += '\n' + label
        plot_fig(output, dx, E, B, cutoff, energies, alpha0, text, xlim, ylim)

def main():
    common.setup_plot()

    output = PdfPages('quadrupole-trajectory-diagram.pdf')
    plot_quadrupole(output)
    output.close()

    output = PdfPages('sextupole-trajectory-diagram.pdf')
    plot_sextupole(output)
    output.close()

if __name__ == '__main__':
    sys.exit(main())
