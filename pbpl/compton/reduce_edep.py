# -*- coding: utf-8 -*-
import os, sys, random
import time
import argparse
import numpy as np
import toml
import tqdm
import pbpl.compton as compton
from pbpl import compton
import Geant4 as g4
from Geant4.hepunit import *
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from collections import namedtuple
import itertools
from functools import reduce
import operator
from scipy.spatial.transform import Rotation

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Project edep files into histograms',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-reduce-edep reduce-edep.toml
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

IndexType = namedtuple('IndexType', 'label unit vals is_binned')

def index_length(index):
    if index.is_binned:
        return len(index.vals)-1
    else:
        return len(index.vals)

def process_index(conf):
    label = conf['Label']
    unit = conf.get('Unit')
    if 'Vals' in conf:
        vals = conf['Vals']
        is_binned = False
    elif 'NumBins' in conf:
        num_bins = conf['NumBins']
        lower_edge = conf['LowerEdge']
        upper_edge = conf['UpperEdge']
        vals = np.linspace(lower_edge, upper_edge, num_bins+1)
        is_binned = True
    else:
        raise ValueError
    return IndexType(label, unit, vals, is_binned)

def nested_get(dictionary, *keys):
    try:
        return reduce(operator.getitem, keys, dictionary)
    except KeyError:
        return None

def project_histogram(filename, conf, indices):
    fin = h5py.File(filename, 'r')
    edep = fin['edep'][:]*keV
    pos = fin['position'][:]*mm
    num_events = fin['edep'].attrs['num_events']
    M = compton.build_transformation(conf, mm, deg)
    tpos = np.array([compton.transform(M, x) for x in pos])
    # from pyevtk.hl import pointsToVTK
    # pointsToVTK("tpos", *np.ascontiguousarray(tpos.T), data=None)
    H, edges = np.histogramdd(
        tpos, tuple(x.vals for x in indices),
        weights = (edep/num_events))
    return H, edges

def main():
    args = get_args()
    conf = args.conf

    fout = h5py.File(conf['Output']['Filename'], 'w')
    gout = fout.require_group(conf['Output']['Group'])

    indices = [process_index(x) for x in conf['Indices']]

    shape = [index_length(x) for x in indices]
    A = np.zeros(shape, dtype=np.float32)

    histo_indices = list(itertools.product(*[range(x) for x in shape[:-3]]))

    fmt = ('{desc:>30s}:{percentage:3.0f}% ' +
           '|{bar}| {n_fmt:>9s}/{total_fmt:<9s}')
    bar = tqdm.tqdm(
        total=len(histo_indices), bar_format=fmt)
    for i in histo_indices:
        bar.update(1)
        histo_index = (str(x) for x in i)
        filename = nested_get(conf['Input'], *histo_index)
        if filename is None:
            continue
        bar.set_description_str(os.path.basename(filename))
        histo, edges = project_histogram(
            filename, conf['Transformation'], indices[-3:])
        A[i] = histo

    gout['edep'] = A/eV
    for i in range(len(indices)-3):
        if isinstance(indices[i].vals[0], str):
            converter = np.string_
        else:
            converter = np.array
        dset_name = 'i{}'.format(i)
        gout[dset_name] = converter(indices[i].vals)
        gout[dset_name].attrs.create('label', np.string_(indices[i].label))
        gout[dset_name].attrs.create('unit', np.string_(indices[i].unit))
    for axis, i in zip(['x', 'y', 'z'], [-3, -2, -1]):
        dset_name = axis + 'bin'
        gout[dset_name] = np.array(indices[i].vals/mm)
        gout[dset_name].attrs.create('unit', np.string_('mm'))

    return 0

if __name__ == '__main__':
    sys.exit(main())
