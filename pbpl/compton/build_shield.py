# -*- coding: utf-8 -*-
import os, sys
import argparse
import numpy as np
import toml
from OCC.Core.BRepAlgoAPI import *
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.BRepMesh import *
from OCC.Core.BRepPrimAPI import *
from OCC.Core.BRepTools import *
from OCC.Core.gp import *
from OCC.Core.STEPControl import *
from OCC.Core.StlAPI import *
from Geant4.hepunit import *
import pbpl.common as common
import pbpl.compton as compton

# Use mm internally for length scale.  OpenCascade internally also uses mm=1.0,
# so don't bother converting lengths when sending/receiving from OpenCascade.
assert(mm == 1.0)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Build scintillator shield STEP model',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-compton-build-shield build-shield.toml
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

def build_slab(conf):
    volume = np.array(conf['Slab']['Volume'])*mm
    slab = BRepPrimAPI_MakeBox(
        gp_Pnt(*volume.T[0]), gp_Pnt(*volume.T[1])).Shape()
    M = compton.build_transformation(conf['Slab']['Transformation'], mm, deg)
    transform = gp_Trsf()
    transform.SetValues(*M.flatten()[:12])
    return BRepBuilderAPI_Transform(slab, transform).Shape()

def build_chamfer(conf):
    x0 = conf['Chamfer']['x0']*mm
    result = BRepPrimAPI_MakeBox(
        gp_Pnt(0.0, -1000.0, -1000.0),
        gp_Pnt(x0, 1000.0, 1000.0))
    return result.Shape()

def build_shield(conf):
    result = build_slab(conf)
    if 'Chamfer' in conf:
        chamfer = build_chamfer(conf)
        result = BRepAlgoAPI_Cut(result, chamfer).Shape()
    return result

def main():
    args = get_args()
    conf = args.conf

    shield = build_shield(conf)

    for outconf in conf['Output']:
        filename = outconf['Filename']
        path = os.path.dirname(filename)
        if path != '':
            os.makedirs(path, exist_ok=True)
        if outconf['Type'] == 'STL':
            # clear out any existing mesh
            breptools_Clean(shield)
            mesh = BRepMesh_IncrementalMesh(
                shield, outconf['LinearDeflection']*mm,
                outconf['IsRelative'], outconf['AngularDeflection']*deg, True)
            stl_writer = StlAPI_Writer()
            stl_writer.SetASCIIMode(False)
            stl_writer.Write(shield, filename)
        elif outconf['Type'] == 'STEP':
            step_writer = STEPControl_Writer()
            step_writer.Transfer(shield, STEPControl_AsIs)
            status = step_writer.Write(filename)

    return 0

if __name__ == '__main__':
    sys.exit(main())
