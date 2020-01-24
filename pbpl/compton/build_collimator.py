# -*- coding: utf-8 -*-
import os, sys
import argparse
import numpy as np
import toml
from collections import namedtuple
import h5py
import scipy.spatial as spatial
from scipy.optimize import minimize_scalar
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import interp1d
from scipy.linalg import norm
from OCC.Core.BRep import *
from OCC.Core.BRepAlgoAPI import *
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.BRepFeat import *
from OCC.Core.BRepMesh import *
from OCC.Core.BRepPrimAPI import *
from OCC.Core.GC import *
from OCC.Core.GeomAPI import *
from OCC.Core.gp import *
from OCC.Core.Interface import *
from OCC.Core.STEPControl import *
from OCC.Core.StlAPI import *
from OCC.Core.TColgp import *
from OCC.Core.TopoDS import *
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from Geant4.hepunit import *
import pbpl.common as common

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
    p0 = np.array(conf['Dimensions']['p0'])*mm
    p1 = np.array(conf['Dimensions']['p1'])*mm
    slab0 = BRepPrimAPI_MakeBox(gp_Pnt(*p0), gp_Pnt(*p1)).Shape()

    translation = np.array(conf['CollimatorTransformation']['Translation'])*mm
    rotation = np.array(conf['CollimatorTransformation']['Rotation'])*deg

    quaternion = gp_Quaternion()
    quaternion.SetEulerAngles(gp_Extrinsic_ZYZ, *rotation)
    transform = gp_Trsf()
    transform.SetTransformation(quaternion, gp_Vec(*translation))
    return BRepBuilderAPI_Transform(slab0, transform).Shape()

def gen_lattice_points(a, xlim, ylim):
    a1 = a*np.array((1.0, 0))
    a2 = a*np.array((0.5, 0.5*np.sqrt(3)))
    points = []
    for i in range(-10, 200):
        for j in range(-100, 100):
            points.append(i*a1 + j*a2)
    points = np.array(points)
    xin_mask = np.logical_and(points[:,0]>=xlim[0], points[:,0]<=xlim[1])
    yin_mask = np.logical_and(points[:,1]>=ylim[0], points[:,1]<=ylim[1])
    mask = np.logical_and(xin_mask, yin_mask)
    return points[mask]

def calculate_pole_mask(conf, lattice, r):
    b1 = conf['Dimensions']['pole_b1']*mm
    c0 = conf['Dimensions']['pole_c0']*mm
    a0 = c0*(3*(b1/c0)**2-1)**(1/3)
    def min_distance_to_curve(p):
        def fmin(y):
            p0 = p[1:3]
            p1 = np.array((y, np.sqrt((y**3+a0**3)/(3*y))))
            return norm(p0-p1)
        result = minimize_scalar(
            fmin, bounds=(5*mm, 125*mm), method='bounded').fun
        return result
    min_distance_to_curve = np.vectorize(
        min_distance_to_curve, signature='(3)->()')
    return min_distance_to_curve(lattice) > r

def build_pores(conf, homing_map, hull):
    a = conf['Pores']['a']*mm
    r = conf['Pores']['r']*a
    z0 = conf['Pores']['z0']*mm
    lz = conf['Pores']['lz']*mm
    ly = conf['Pores']['ly']*mm
    p0 = np.array(conf['Dimensions']['p0'])*mm

    zy_lattice = gen_lattice_points(a, (0, lz), (0, 0.5*ly))
    zy_lattice += np.array((z0, 0))
    # zy --> xyz
    lattice = np.hstack(
        (p0[0]*np.ones((len(zy_lattice), 1)), np.flip(zy_lattice, axis=1)))

    # retain only lattice points within convex hull of CST trajectories
    delaunay = spatial.Delaunay(hull.points)
    lattice = lattice[delaunay.find_simplex(lattice[:,1:3])>=0]
    homing = homing_map(lattice[:,1:3])

    # transform lattice to collimator's entrance plane
    translation = np.array(conf['CollimatorTransformation']['Translation'])*mm
    euler = np.array(conf['CollimatorTransformation']['Rotation'])*deg
    rotation = spatial.transform.Rotation.from_euler('zyz', euler).as_dcm()
    lattice = (rotation @ lattice.T).T + translation

    pole_mask = calculate_pole_mask(conf, lattice, a)
    lattice = lattice[pole_mask]
    homing = homing[pole_mask]

    cos_theta = homing.T[0]
    phi = homing.T[1]

    mask = cos_theta>0.6
    lattice = lattice[mask]
    cos_theta = cos_theta[mask]
    sin_theta = np.sqrt(1-cos_theta**2)
    phi = phi[mask]

    # yzx -> xyz
    nhat = np.array(
        (cos_theta, sin_theta * np.cos(phi), sin_theta * np.sin(phi))).T

    nhat = (rotation @ nhat.T).T
    nscint = (rotation @ np.array((1,0,0)))

    reflected_lattice = []
    reflected_nhat = []
    for i, (p, n) in enumerate(zip(lattice, nhat)):
        if np.abs(p[1]) < 0.1*a:
            p[1] = 0.0
            n[1] = 0.0
            lattice[i] = p
            nhat[i] = n
            continue
        reflected_lattice.append(np.array((p[0], -p[1], p[2])))
        reflected_nhat.append(np.array((n[0], -n[1], n[2])))
    lattice = np.concatenate((lattice, np.array(reflected_lattice)))
    nhat = np.concatenate((nhat, np.array(reflected_nhat)))

    cutter = TopoDS_CompSolid()
    bilda = BRep_Builder()
#    cutter = TopoDS_Compound()
#    bilda.MakeCompound(cutter)
    bilda.MakeCompSolid(cutter)

    for p, n in zip(lattice, nhat):
        circle = GC_MakeCircle(
            gp_Ax2(gp_Pnt(*(p-n*20*mm)), gp_Dir(*nscint)), r)
        edge = BRepBuilderAPI_MakeEdge(circle.Value()).Edge()
        wire = BRepBuilderAPI_MakeWire(edge)
        face = BRepBuilderAPI_MakeFace(wire.Wire())
        cylinder = BRepPrimAPI_MakePrism(face.Shape(), gp_Vec(*(n*50*mm)))
        bilda.Add(cutter, cylinder.Shape())
    return cutter


def build_upperpole(conf):
    b1 = conf['Dimensions']['pole_b1']*mm
    c0 = conf['Dimensions']['pole_c0']*mm

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

def build_pole(conf):
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

def build_collimator(conf, homing_map, hull):
    slab = build_slab(conf)
    pole = build_pole(conf)
    pores = build_pores(conf, homing_map, hull)
    cut_slab = BRepAlgoAPI_Cut(slab, pole).Shape()
    cut_pores = BRepAlgoAPI_Cut(cut_slab, pores).Shape()
    return cut_pores

def scale_limits(ax, scale):
    lim = ax.get_view_interval()
    mean = 0.5*(lim[0]+lim[1])
    width = lim[1]-lim[0]
    new_width = scale*width
    ax.set_view_interval(mean-0.5*new_width, mean+0.5*new_width)

def get_energy(gin):
    m0 = gin['m0'][()]*kg
    p0 = gin['p'][0]*m0*c_light
    E0 = np.sqrt(norm(p0)**2*c_light**2 + m0**2*c_light**4)
    KE = E0 - m0*c_light**2
    return KE

TrajectoryMapping = namedtuple(
    'TrajectoryMapping', 'group energy y0 y1 ymap zmap xi phi')

def analyze_trajectories(conf, fin):
    result = []
    translation = -np.array(
        conf['ScintillatorTransformation']['Translation'])*mm
    euler = -np.array(conf['ScintillatorTransformation']['Rotation'])*deg
    rotation = spatial.transform.Rotation.from_euler('zyz', euler).as_dcm()
    p0 = np.array(conf['Dimensions']['p0'])*mm
    p1 = np.array(conf['Dimensions']['p1'])*mm
    for group_name, gin in fin.items():
        energy = get_energy(gin)
        x = gin['x'][:]*meter

        # transform from scintillator entrance plane to YZ plane
        xt = (rotation @ (x+translation).T).T

        # CST trajectories penetrate about 100um into scintillator
        if np.abs(xt[-1,0]-1*mm)>1*mm:
            # These trajectories terminated outside of the scintillator
            # entrance plane.  Mostly, these were low energy trajectories
            # that deflected strongly without hitting scintillator.
            continue

        i1 = np.argmax(xt[:,0]>p0[0])
        assert(i1 != 0)

        i0 = i1-1
        xit = np.concatenate(
            ([p0[0]], interp1d(xt[i0:i1+1,0], xt[i0:i1+1,1:3].T)(p0[0])))
        xi = (rotation.T @ xit) - translation
        xf = x[-1]
        xft = xt[-1]
        nhat = xft-xit
        nhat = nhat / norm(nhat)
        # xyz -> yzx
        xi = nhat[0]
        phi = np.arctan2(nhat[2], nhat[1])
        result.append(
            TrajectoryMapping(
                group=group_name, energy=energy, y0=x[0,1], y1=x[-1,1],
                ymap=xit[1], zmap=xit[2], xi=xi, phi=phi))
    return result

def create_homing_map(mappings):
    energy_map = {}
    for m in mappings:
        if m.energy not in energy_map:
            energy_map[m.energy] = []
        energy_map[m.energy].append(m)

    for m in energy_map.values():
        A = np.array([x.y1 for x in m])
        diff = A[1:] - A[:-1]
        i_decrease = np.argmax(diff < 0)
        if i_decrease == 0:
            continue
        del m[i_decrease+1:]

    position = []
    data = []
    for m in mappings:
        position.append(np.array((m.ymap, m.zmap)))
        data.append(np.array((m.xi, m.phi)))
    position = np.array(position)
    data = np.array(data)
    linear_interp = LinearNDInterpolator(position, data)
    nearest_interp = NearestNDInterpolator(position, data)
    hull = spatial.ConvexHull(position)

    mesh_dx = 0.5*mm
    sigma_mesh = 5*mm

    Y, Z = np.meshgrid(
        np.arange(
            hull.min_bound[0]-mesh_dx, hull.max_bound[0]+mesh_dx, mesh_dx),
        np.arange(
            hull.min_bound[1]-mesh_dx, hull.max_bound[1]+mesh_dx, mesh_dx))
    A = np.array((Y, Z))
    A = np.rollaxis(A, 0, 3)

    xi = linear_interp(A)[:,:,0]
    mask = np.isnan(xi)
    xi[mask] = nearest_interp(A)[mask,0]
    smoothed_xi = gaussian_filter(xi, sigma_mesh/mesh_dx, mode='nearest')

    phi = linear_interp(A)[:,:,1]
    phi[mask] = nearest_interp(A)[mask,1]
    smoothed_phi = gaussian_filter(phi, sigma_mesh/mesh_dx, mode='nearest')

    smooth_interp = LinearNDInterpolator(
        np.array((Y.ravel(), Z.ravel())).T,
        np.array((smoothed_xi.ravel(), smoothed_phi.ravel())).T)

    return smooth_interp, hull


def plot_homing_map(filename, homing_map, hull):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    output = PdfPages(filename)
    common.setup_plot()
    plot.rc('figure.subplot', right=0.99, top=0.99, bottom=0.09, left=0.10)

    mesh_dx = 0.5*mm

    Y, Z = np.meshgrid(
        np.arange(
            hull.min_bound[0]-mesh_dx, hull.max_bound[0]+mesh_dx, mesh_dx),
        np.arange(
            hull.min_bound[1]-mesh_dx, hull.max_bound[1]+mesh_dx, mesh_dx))
    smoothed_xi = homing_map(Y, Z)[:,:,0]
    smoothed_phi = homing_map(Y, Z)[:,:,1]

    fig = plot.figure(figsize=(244/72, 100/72))
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)
    ax.contour(
        Z/mm, Y/mm, smoothed_xi, levels=np.linspace(-1.0, 1.0, 20),
        linewidths=0.3, colors='#0083b8',
        linestyles='solid', zorder=0)
    ax.fill(
        hull.points[hull.vertices,1]/mm,
        hull.points[hull.vertices,0]/mm,
        fill=False, linewidth=0.5, edgecolor='k', zorder=10)
    scale_limits(ax.xaxis, 1.1)
    scale_limits(ax.yaxis, 1.1)
    plot.title(r'$\xi(z, y)$')
    plot.xlabel(r'$z_{\rm scint}$ (mm)', labelpad=0.0)
    plot.ylabel(r'$y_{\rm scint}$ (mm)', labelpad=0.0)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    output.savefig(fig, transparent=True)

    fig = plot.figure(figsize=(244/72, 100/72))
    ax = fig.add_subplot(1, 1, 1, aspect=1.0)
    ax.contour(
        Z/mm, Y/mm, smoothed_phi, levels=20,
        linewidths=0.3, colors='#0083b8',
        linestyles='solid', zorder=0)
    ax.fill(
        hull.points[hull.vertices,1]/mm,
        hull.points[hull.vertices,0]/mm,
        fill=False, linewidth=0.5, edgecolor='k', zorder=10)
    scale_limits(ax.xaxis, 1.1)
    scale_limits(ax.yaxis, 1.1)
    plot.title(r'$\phi(z, y)$')
    plot.xlabel(r'$z_{\rm scint}$ (mm)', labelpad=0.0)
    plot.ylabel(r'$y_{\rm scint}$ (mm)', labelpad=0.0)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    output.savefig(fig, transparent=True)
    output.close()


def main():
    args = get_args()
    conf = args.conf

    with h5py.File(conf['Files']['Input'], 'r') as fin:
        mappings = analyze_trajectories(conf, fin)
        homing_map, hull = create_homing_map(mappings)
        if 'DiagnosticOutput' in conf['Files']:
            plot_homing_map(
                conf['Files']['DiagnosticOutput'], homing_map, hull)

    collimator = build_collimator(conf, homing_map, hull)

    if 'STEPOutput' in conf['Files']:
        step_writer = STEPControl_Writer()
        step_writer.Transfer(collimator, STEPControl_AsIs)
        status = step_writer.Write(conf['Files']['STEPOutput'])

    if 'STLOutput' in conf['Files']:
        mesh = BRepMesh_IncrementalMesh(collimator, 1.0, True, 20*deg, True)
        stl_writer = StlAPI_Writer()
        stl_writer.SetASCIIMode(False)
        stl_writer.Write(collimator, conf['Files']['STLOutput'])

    return 0

if __name__ == '__main__':
    sys.exit(main())
