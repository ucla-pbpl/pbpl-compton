#!/usr/bin/env python
import os, sys
import numpy as np
import lmfit
from scipy.linalg import norm
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from pbpl import common
from pbpl import compton
from pbpl.common.units import *
from num2tex import num2tex

def get_energy(gin):
    m0 = gin['m0'][()]*kg
    p0 = gin['p'][0]*m0*c_light
    E0 = np.sqrt(norm(p0)**2*c_light**2 + m0**2*c_light**4)
    KE = E0 - m0*c_light**2
    return KE

def fit_func(x, c0, c1, c2):
    return np.exp(c0 + c1*x + c2*x**2)

def main():
    common.setup_plot()

    energy = []
    position = []
    theta0 = 28.0*deg

    M = compton.build_transformation(
        [ ['TranslateX', 'RotateY', 'TranslateZ'],
          [-40.1, -28.0, -30.0] ], mm, deg)

    with h5py.File('trajectories/trajectories.h5', 'r') as fin:
        for gin in fin.values():
            E0 = get_energy(gin)
            x0 = gin['x'][0]*meter
            x1 = gin['x'][-1]*meter
            if x0[1] != 0.0:
                continue
            x1 = compton.transform(M, x1)
            energy.append(E0)
            position.append(x1[2])
    energy = np.array(energy)
    position = np.array(position)
    args = np.argsort(energy)
    energy = energy[args]
    position = position[args]

    mod = lmfit.Model(fit_func)
    params = mod.make_params(c0=0.0, c1=0.0, c2=0.0)

    result = mod.fit(data=energy/MeV, x=position, params=params)
    print(result.fit_report())

    v = result.params.valuesdict()
    print(v['c0'])
    print(v['c1'])
    print(v['c2'])

    x_fit = np.linspace(position[0], position[-1], 200)

    fig = plot.figure(figsize=(244.0/72, 120.0/72))
    ax = fig.add_subplot(1, 1, 1)


    ax.semilogy(
        x_fit/mm, result.eval(x=x_fit), linewidth=0.6)

    ax.semilogy(
        position/mm, energy/MeV, marker='.', ls='', markersize=0.001,
        color='k')

    ax.text(
        0.05, 0.8,
        r'$E(z_s)/{\rm MeV} = \exp (c_0 + c_1 z_s + c_2 z_s^2)$',
        fontsize=7.0,
        transform=ax.transAxes)

    text = r'$c_0 = {:.3}$'.format(v['c0']) + '\n'
    text += r'$c_1 = {:.3}'.format(num2tex(v['c1'] * mm))
    text += r'\;{\rm mm}^{-1}$' + '\n'
    text += r'$c_2 = {:.3}'.format(num2tex(v['c2'] * mm**2))
    text += r'\;{\rm mm}^{-2}$' + '\n'
    print(text)
    ax.text(0.5, 0.2, text, transform=ax.transAxes)

    ax.set_xlabel(r'$z_s$ (mm)', labelpad=-1.0)
    ax.set_ylabel(r"Energy (MeV)", labelpad=2.0)
    # ax.set_xlim(0, 250)
    ax.set_ylim(0.1, 20)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    filename = 'out/energy-scale.pdf'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plot.savefig(filename, transparent=True)

if __name__ == '__main__':
    sys.exit(main())
