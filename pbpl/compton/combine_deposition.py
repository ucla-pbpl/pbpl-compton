# -*- coding: utf-8 -*-
import os, sys, random
import argparse
import numpy as np
import toml
import asteval
from pbpl import compton
# import Geant4 as g4
# from Geant4.hepunit import *
import h5py
import pbpl.common as common
from pbpl.common.units import *
from collections import namedtuple

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Combine energy deposition',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-compton-combine-deposition combine-deposition.toml
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

def get_input(conf):
    edep = {}
    for c in conf:
        with h5py.File(c['Filename'], 'r') as fin:
            run_index = tuple(c['RunIndex'])
            _num_events = fin['num_events'][run_index]
            gin = fin[c['Group']]
            _edep = gin['edep'][run_index]*MeV
            _xbin = gin['xbin'][:]*mm
            _ybin = gin['ybin'][:]*mm
            _zbin = gin['zbin'][:]*mm
            if len(edep) == 0:
                xbin = _xbin
                ybin = _ybin
                zbin = _zbin
                num_events = _num_events
            else:
                assert(np.array_equal(xbin, _xbin))
                assert(np.array_equal(ybin, _ybin))
                assert(np.array_equal(zbin, _zbin))
                assert(num_events == _num_events)
            edep[c['Key']] = _edep
    return edep, xbin, ybin, zbin, num_events

def main():
    args = get_args()
    conf = args.conf
    edep, xbin, ybin, zbin, num_events = get_input(conf['Input'])

    with h5py.File(conf['Output']['Filename'], 'w') as fout:
        fout['num_events'] = np.array((num_events,))
        fout['i0'] = np.array((np.string_('yo'),))
        if 'Group' in conf['Output']:
            gout = fout.create_group(conf['Output']['Group'])
        else:
            gout = fout
        gout['edep'] = ((edep['A'] + edep['B'])/MeV).astype('float32')[np.newaxis,:]
        gout['edep'].attrs.create('num_events', num_events)
        gout['edep'].attrs.create('unit', np.string_('MeV'))
        gout['xbin'] = xbin/mm
        gout['ybin'] = ybin/mm
        gout['zbin'] = zbin/mm
        for dset_name in ['xbin', 'ybin', 'zbin']:
            gout[dset_name].attrs.create('unit', np.string_('mm'))
        fout.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())
