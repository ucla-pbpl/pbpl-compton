#!/usr/bin/env python
import os, sys
import argparse
import toml
import asteval
from collections import namedtuple
import math
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
from functools import reduce

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Calculate energy scale from CST trajectory map',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-compton-calc-energy-scale calc-energy-scale.toml
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

def get_energy(gin):
    m0 = gin['m0'][()]*kg
    p0 = gin['p'][0]*m0*c_light
    E0 = np.sqrt(norm(p0)**2*c_light**2 + m0**2*c_light**4)
    KE = E0 - m0*c_light**2
    return KE

def fit_func(x, c0, c1, c2, c3):
    return c0 + c1*x + c2*x**2 + c3*x**3

def in_volume(vol, x):
    return np.logical_and.reduce(
        (x[:,0]>=vol[0,0], x[:,0]<=vol[0,1],
         x[:,1]>=vol[1,0], x[:,1]<=vol[1,1],
         x[:,2]>=vol[2,0], x[:,2]<=vol[2,1]))

Axis = namedtuple('Axis', 'label unit xlim')

def get_axis(aeval, label, unit, xlim):
    xlim = aeval(xlim)
    if xlim is not None:
        xlim = np.array(xlim)
    return Axis(label, aeval(unit), xlim)

def plot_annotation(ax, aeval, conf):
    if 'Annotation' in conf:
        for aconf in conf['Annotation']:
            text = ''
            for s in aconf['Text']:
                text += aeval(s) + '\n'
            kwargs = {}
            if 'Size' in aconf:
                kwargs['size'] = aconf['Size']
            ax.text(
                *aconf['Location'], text, va='top',
                transform=ax.transAxes, **kwargs)

def main():
    args = get_args()
    conf = args.conf

    # create safe interpreter for evaluation of configuration expressions
    aeval = asteval.Interpreter(use_numpy=True)
    for q in common.units.__all__:
        aeval.symtable[q] = common.units.__dict__[q]

    pconf = conf['Projection']
    M = compton.build_transformation(pconf['Transformation'], mm, deg)
    prefilter = np.array(pconf['Prefilter'])*mm
    postfilter = np.array(pconf['Postfilter'])*mm

    energy = []
    position = []

    x = []
    E0 = []
    with h5py.File(conf['Files']['Input'], 'r') as fin:
        for gin in fin.values():
            x.append(
                (gin['x'][0]*meter, compton.transform(M, gin['x'][-1]*meter)))
            E0.append(get_energy(gin))
    x = np.array(x)
    E0 = np.array(E0)
    prefilter_mask = in_volume(prefilter, x[:,0,:])
    x_pre = x[prefilter_mask,:,:]
    E0_pre = E0[prefilter_mask]
    postfilter_mask = in_volume(postfilter, x_pre[:,1,:])
    x_post = x_pre[postfilter_mask,:,:]
    E0_post = E0_pre[postfilter_mask]

    energy = E0_post.copy()
    position = x_post[:,1,2].copy()
    args = np.argsort(energy)
    energy = energy[args]
    position = position[args]

    mod = lmfit.Model(fit_func)
    params = mod.make_params(c0=0.0, c1=0.0, c2=0.0, c3=0.0)

    result = mod.fit(
        data=np.log(energy/MeV), x=position, params=params)

    v = result.params.valuesdict()
    x_fit = np.linspace(position[0], position[-1], 200)

    common.setup_plot()

    fig = plot.figure(figsize=np.array(conf['Plot']['FigSize'])/72)
    ax = fig.add_subplot(1, 1, 1)

    axes = [get_axis(aeval, *conf['Plot'][x]) for x in ['XAxis', 'YAxis']]

    ax.semilogy(
        x_fit/axes[0].unit, np.exp(result.eval(x=x_fit)), linewidth=0.6)

    ax.semilogy(
        position/axes[0].unit, energy/axes[1].unit,
        marker='.', ls='', markersize=2.0, markeredgewidth=0,
        color='k')

    aeval.symtable['fitval'] = v
    aeval.symtable['num2tex'] = num2tex
    plot_annotation(ax, aeval, conf['Plot'])

    ax.set_xlabel(axes[0].label, labelpad=-1.0)
    ax.set_ylabel(axes[1].label, labelpad=2.0)

    ax.set_xlim(*axes[0].xlim)
    ax.set_ylim(*axes[1].xlim)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    filename = conf['Files']['PlotOutput']
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plot.savefig(filename, transparent=True)

    if 'CalcOutput' in conf['Files']:
        filename = conf['Files']['CalcOutput']
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        calc_output = {
            'EnergyScaleCoefficients' :
            { 'c0' : float(v['c0']),
              'c1' : float(v['c1']*mm),
              'c2' : float(v['c2']*mm**2),
              'c3' : float(v['c3']*mm**3) } }
        with open(filename, 'w') as fout:
            toml.dump(calc_output, fout)

if __name__ == '__main__':
    sys.exit(main())
