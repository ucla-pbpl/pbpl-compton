# -*- coding: utf-8 -*-
import os, sys
import argparse
import numpy as np
import toml
import h5py
import scipy.spatial as spatial
from scipy.optimize import minimize_scalar
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import ode
from scipy.linalg import norm, inv
from scipy.ndimage.interpolation import shift
from shapely.geometry import LinearRing
from OCC.Core.BRep import *
from OCC.Core.BRepAlgoAPI import *
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.BRepMesh import *
from OCC.Core.BRepOffsetAPI import *
from OCC.Core.BRepPrimAPI import *
from OCC.Core.BRepTools import *
from OCC.Core.GC import *
from OCC.Core.GeomAPI import *
from OCC.Core.gp import *
from OCC.Core.STEPControl import *
from OCC.Core.StlAPI import *
from OCC.Core.TColgp import *
from OCC.Core.TopoDS import *
from scipy.ndimage import gaussian_filter
from Geant4.hepunit import *
import pbpl.common as common
import pbpl.compton as compton

# Use mm internally for length scale.  OpenCascade internally also uses mm=1.0,
# so don't bother converting lengths when sending/receiving from OpenCascade.
assert(mm == 1.0)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Build collimator STEP model',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-compton-build-collimator build-collimator.toml
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

def collect(items):
    result = TColgp_Array1OfPnt(0, len(items)-1)
    for i, item in enumerate(items):
        result.SetValue(i, item)
    return result

def build_slab(conf):
    volume = np.array(conf['Slab']['Volume'])*mm
    slab = BRepPrimAPI_MakeBox(
        gp_Pnt(*volume.T[0]), gp_Pnt(*volume.T[1])).Shape()
    M = compton.build_transformation(conf['Slab']['Transformation'], mm, deg)
    transform = gp_Trsf()
    transform.SetValues(*M.flatten()[:12])
    return BRepBuilderAPI_Transform(slab, transform).Shape()

def build_upperpole(conf):
    b1 = conf['Slab']['magnet_b1']*mm
    c0 = conf['Slab']['magnet_c0']*mm

    a0 = c0*(3*(b1/c0)**2-1)**(1/3)
    y = np.arange(c0-1*mm, 126.001*mm, 1*mm)
    z = np.sqrt((y**3+a0**3)/(3*y))
    bspline = GeomAPI_PointsToBSpline(collect(
        [gp_Pnt(0, y_, z_) for y_, z_ in zip(y,z)])).Curve()

    p0 = gp_Pnt(0, y[0], z[0])
    p1 = gp_Pnt(0, y[-1], z[-1])
    p2 = gp_Pnt(0, y[-1], z[0])
    s12 = GC_MakeSegment(p1, p2).Value()
    s20 = GC_MakeSegment(p2, p0).Value()
    wire = BRepBuilderAPI_MakeWire(
        BRepBuilderAPI_MakeEdge(bspline).Edge(),
        BRepBuilderAPI_MakeEdge(s12).Edge(),
        BRepBuilderAPI_MakeEdge(s20).Edge())
    face = BRepBuilderAPI_MakeFace(wire.Wire())
    extrusion = BRepPrimAPI_MakePrism(
        face.Shape(), gp_Vec(1000, 0, 0))

    return extrusion.Shape()

def build_magnet(conf):
    upper = build_upperpole(conf)
    ymirror = gp_Trsf()
    ymirror.SetMirror(gp_ZOX())
    lower = BRepBuilderAPI_Transform(upper, ymirror).Shape()
    pole = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(pole)
    builder.Add(pole, upper)
    builder.Add(pole, lower)
    return pole


def build_pores(conf, lattice):
    cutter = TopoDS_Compound()
    bilda = BRep_Builder()
    bilda.MakeCompound(cutter)

    grid = np.array(conf['Slab']['Volume'])*mm
    x_cs = np.linspace(grid[0,0], grid[0,1], conf['Pores']['NumCrossSections'])
    for i, q in enumerate(lattice):
        print(i, len(lattice))
        pore = BRepOffsetAPI_ThruSections(True, False)
        for x in x_cs:
            # BRepOffsetAPI_MakeOffsetShape is suuuuuuuper sloooooooow.
            # Instead, use Shapely to shrink pore cross section.
            ring = []
            for trajectory in q:
                ring.append(trajectory(x))
            ring = LinearRing(ring)
            pore_coords = np.array(
                ring.parallel_offset(0.25*mm, 'right', join_style=2))[:-1]

            polygon = BRepBuilderAPI_MakePolygon()
            for coord in pore_coords:
                polygon.Add(gp_Pnt(x, *coord))
            polygon.Close()
            pore.AddWire(polygon.Wire())
        pore.Build()
        if pore.Shape() is not None:
            bilda.Add(cutter, pore.Shape())

    M = compton.build_transformation(conf['Slab']['Transformation'], mm, deg)
    transform = gp_Trsf()
    transform.SetValues(*M.flatten()[:12])
    return BRepBuilderAPI_Transform(cutter, transform).Shape()

def build_chamfer(conf):
    x0 = conf['Chamfer']['x0']*mm
    result = BRepPrimAPI_MakeBox(
        gp_Pnt(0.0, -1000.0, 0.0),
        gp_Pnt(x0, 1000.0, 1000.0))
    return result.Shape()

def build_collimator(conf, lattice):
    slab = build_slab(conf)
    magnet = build_magnet(conf)
    result = BRepAlgoAPI_Cut(slab, magnet).Shape()
    pores = build_pores(conf, lattice)
    result = BRepAlgoAPI_Cut(result, pores).Shape()
    if 'Chamfer' in conf:
        chamfer = build_chamfer(conf)
        result = BRepAlgoAPI_Cut(result, chamfer).Shape()
    return result

def calc_vectorfield(conf):
    M = compton.build_transformation(conf['Slab']['Transformation'], mm, deg)
    M = inv(M)
    grid = np.array(conf['Slab']['Volume'])*mm
    dx = np.array(conf['VectorField']['GridResolution'])*mm

    result = []
    with h5py.File(conf['VectorField']['Input'], 'r') as fin:
        for group_name, gin in fin.items():
            x = gin['x'][:]*meter
            Mx = np.array([compton.transform(M, q) for q in x])
            grid_mask = compton.in_volume(grid, Mx)
            grid_mask += shift(grid_mask, -10) + shift(grid_mask, 1)
            Mx = Mx[grid_mask].copy()
            delta = np.diff(Mx, axis=0)
            n = delta/norm(delta, axis=1, keepdims=True)
            result.append(np.array((Mx[:-1], n)))
    x, n = np.hstack(result)
    mirror_x = x * np.array((1,-1,1))
    delaunay = spatial.Delaunay(np.concatenate((x, mirror_x)))

    nearest_interp = NearestNDInterpolator(x, n)
    linear_interp = LinearNDInterpolator(x, n)
    epsilon = 0.1*dx
    dims = [np.arange(grid[q,0], grid[q,1]+epsilon, dx) for q in [0,1,2]]

    # Use linear interpolation within convex hull, else use nearest-neighbor.
    X, Y, Z = np.meshgrid(*dims, indexing='ij')
    n_XYZ = linear_interp((X, np.abs(Y), Z))
    n_XYZ_nearest = nearest_interp((X, np.abs(Y), Z))
    mask = np.isnan(n_XYZ)
    n_XYZ[mask] = n_XYZ_nearest[mask]
    n_XYZ[Y<0] *= np.array((1,-1,1))

    if 'DiagnosticOutput' in conf['VectorField']:
        with h5py.File(conf['VectorField']['DiagnosticOutput'], 'w') as fout:
            fout['nx'] = n_XYZ[:,:,:,0]
            fout['ny'] = n_XYZ[:,:,:,1]
            fout['nz'] = n_XYZ[:,:,:,2]

    interpolator = RegularGridInterpolator(
        dims, n_XYZ, bounds_error=False, fill_value=None)
    return interpolator, delaunay

def gen_lattice_points(a, xlim, ylim):
    a1 = a*np.array((1.0, 0))
    a2 = a*np.array((0.5, 0.5*np.sqrt(3)))
    points = []
    for i in range(-10, 250):
        for j in range(-100, 100):
            points.append(i*a1 + j*a2)
    points = np.array(points)
    xin_mask = np.logical_and(points[:,0]>=xlim[0], points[:,0]<=xlim[1])
    yin_mask = np.logical_and(points[:,1]>=ylim[0], points[:,1]<=ylim[1])
    mask = np.logical_and(xin_mask, yin_mask)
    return points[mask]

def calc_trajectory(interpolator, y0, max_length):
    def yprime(t, y):
        dy = interpolator(y)
        return dy
    solver = ode(yprime)
    solver.set_integrator('dopri5')
    solver.set_initial_value(y0)
    trajectory = []
    dt = 1*mm
    curr_t = 0.0
    while 1:
        curr_t += dt
        trajectory.append(solver.integrate(curr_t))
        if curr_t >= max_length:
            return None
        if solver.y[0] > 0:
            break
    trajectory = np.array(trajectory)
    result = interp1d(
        trajectory.T[0], trajectory.T[1:3],
        bounds_error=False, fill_value='extrapolate')
    return result

def calculate_pole_mask(conf, lattice, r):
    b1 = conf['Slab']['magnet_b1']*mm
    c0 = conf['Slab']['magnet_c0']*mm
    a0 = c0*(3*(b1/c0)**2-1)**(1/3)
    lattice = lattice.copy()
    lattice[:,1] = np.abs(lattice[:,1])

    def below_curve(p):
        y, z = p[1:3]
        if y == 0:
            return True
        else:
            return z<np.sqrt((y**3+a0**3)/(3*y))
    below_curve = np.vectorize(
        below_curve, signature='(3)->()')

    def min_distance_to_curve(p):
        def fmin(y):
            p0 = p[1:3]
            p1 = np.array((y, np.sqrt((y**3+a0**3)/(3*y))))
            return norm(p0-p1)
        result = minimize_scalar(
            fmin, bounds=(1*mm, 150*mm), method='bounded').fun
        return result
    min_distance_to_curve = np.vectorize(
        min_distance_to_curve, signature='(3)->()')

    return np.logical_and(
        below_curve(lattice), min_distance_to_curve(lattice) > r)


def calc_lattice(conf, interpolator, delaunay):
    pore_conf = conf['Pores']
    a = pore_conf['a']*mm
    delta_z = pore_conf['delta_z']*mm
    z0 = pore_conf['z0']*mm
    lz = pore_conf['lz']*mm
    ly = pore_conf['ly']*mm
    magnet_buffer = pore_conf['MagnetBuffer']*mm
    max_length = pore_conf['MaxLength']*mm
    volume = np.array(conf['Slab']['Volume'])*mm
    M = compton.build_transformation(conf['Slab']['Transformation'], mm, deg)

    zy_lattice = gen_lattice_points(a, (0, lz), (-0.5*ly, 0.5*ly))
    zy_lattice += np.array((delta_z, 0))
    # zy --> xyz
    tri_lattice = np.hstack(
        (volume[0,0]*np.ones((len(zy_lattice), 1)),
         np.flip(zy_lattice, axis=1)))

    tri_lattice = tri_lattice[tri_lattice[:,2] > z0]
    tri_lattice = tri_lattice[delaunay.find_simplex(tri_lattice)>=0]
    M_tri_lattice = np.array([compton.transform(M, q) for q in tri_lattice])

    # discard lattice points within a given distance from magnet pole
    tri_lattice = tri_lattice[
        calculate_pole_mask(conf, M_tri_lattice, magnet_buffer)]

    hex_lattice = {}
    N = 6
    phi = (30*deg) + np.linspace(0, 2*np.pi, N, endpoint=False)
    hex_coords = (a/np.sqrt(3))*np.array(
        (np.zeros_like(phi), np.sin(phi), np.cos(phi))).T

    # integrate hexagonal lattice through vectorfield
    for i, p in enumerate(tri_lattice):
        print(i, len(tri_lattice))
        for q in hex_coords:
            coord = p+q
            key = (round(coord[1], 3), round(coord[2], 3))
            if key in hex_lattice:
                continue
            if delaunay.find_simplex(coord) == -1:
                # hex lattice point is outside trajectory hull
                hex_lattice[key] = None
                continue
            trajectory = calc_trajectory(interpolator, coord, max_length)
            if trajectory == None:
                # trajectory didn't cross x=0 plane
                hex_lattice[key] = None
                continue
            hex_lattice[key] = trajectory

    # assemble result
    result = []
    for p in tri_lattice:
        valid_hex = True
        trajectories = []
        for q in hex_coords:
            coord = p+q
            key = (round(coord[1], 3), round(coord[2], 3))
            if hex_lattice[key] == None:
                valid_hex = False
                break
            trajectories.append(hex_lattice[key])
        if valid_hex:
            result.append(trajectories)
    return result


def main():
    args = get_args()
    conf = args.conf

    interpolator, delaunay = calc_vectorfield(conf)
    lattice = calc_lattice(conf, interpolator, delaunay)
    collimator = build_collimator(conf, lattice)

    for outconf in conf['Output']:
        if outconf['Type'] == 'STL':
            # clear out any existing mesh
            breptools_Clean(collimator)
            mesh = BRepMesh_IncrementalMesh(
                collimator, outconf['LinearDeflection']*mm,
                outconf['IsRelative'], outconf['AngularDeflection']*deg, True)
            stl_writer = StlAPI_Writer()
            stl_writer.SetASCIIMode(False)
            stl_writer.Write(collimator, outconf['Filename'])
        elif outconf['Type'] == 'STEP':
            step_writer = STEPControl_Writer()
            step_writer.Transfer(collimator, STEPControl_AsIs)
            status = step_writer.Write(outconf['Filename'])

    return 0

if __name__ == '__main__':
    sys.exit(main())
