# -*- coding: utf-8 -*-
import os, sys, random
import argparse
import asteval
import numpy as np
import toml
import time
from collections import deque
import Geant4 as g4
from Geant4.hepunit import *
import random
from pbpl import compton
import h5py
from importlib import import_module
from collections import namedtuple
from treelib import Node, Tree

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Run Compton Spectrometer Monte Carlo',
        epilog='''\
Example:

.. code-block:: sh

  > pbpl-compton-mc lil-cpt.toml vis.mac''')
    parser.add_argument(
        'config_filename', metavar='TOML',
        help='Monte Carlo configuration file')
    parser.add_argument(
        'macro_filenames', metavar='MAC', nargs='*',
        help='Geant4 macro files to be executed (default=none)')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args.conf = toml.load(args.config_filename)
    return args

class PrimaryGeneratorAction(g4.G4VUserPrimaryGeneratorAction):
    def __init__(self, conf):
        g4.G4VUserPrimaryGeneratorAction.__init__(self)
        self.pg = g4.G4ParticleGun(1)
        self.conf = conf
        c = conf['PrimaryGenerator']
        p, m = c['PythonGenerator'].rsplit('.', 1)
        gen_args = []
        if 'PythonGeneratorArgs' in c:
            for x in c['PythonGeneratorArgs']:
                try:
                    gen_args.append(eval(x))
                except:
                    gen_args.append(x)
        sys.path.append('./')
        mod = import_module(p)
        self.generator = getattr(mod, m)(*gen_args)

    def GeneratePrimaries(self, event):
        try:
            particle_name, position, direction, energy = next(self.generator)
            self.pg.SetParticleByName(particle_name)
            # self.pg.SetParticlePolarization(g4.G4ThreeVector(0, 1, 0))
            self.pg.SetParticlePosition(position)
            self.pg.SetParticleMomentumDirection(direction)
            self.pg.SetParticleEnergy(energy)
            self.pg.GeneratePrimaryVertex(event)
        except StopIteration:
            event.SetEventAborted()

class MyRunAction(g4.G4UserRunAction):
    "My Run Action"

    def BeginOfRunAction(self, run):
        pass

    def EndOfRunAction(self, run):
        pass

class StatusEventAction(g4.G4UserEventAction):
    "Status Event Action"

    def __init__(self, num_events, update_period):
        g4.G4UserEventAction.__init__(self)
        sys.stderr.write('TOT={}\n'.format(num_events))
        sys.stderr.flush()
        self.num_events = num_events
        self.count = 0
        self.update_period = update_period
        self.prev_time = time.time()

    def BeginOfEventAction(self, event):
        pass

    def EndOfEventAction(self, event):
        curr_time = time.time()
        elapsed = curr_time - self.prev_time
        if elapsed >= self.update_period:
            sys.stderr.write('CUR={}\n'.format(self.count))
            sys.stderr.flush()
            self.prev_time = curr_time
        self.count += 1

class MyStackingAction(g4.G4UserStackingAction):
    "My Stacking Action"

    def ClassifyNewTrack(self, track):
        print('ClassifyNewTrack')
        return track

    def NewStage(self):
        print('NewStage')

    def PrepareNewEvent(self):
        print('PrepareNewEvent')

TrackingNode = namedtuple(
    'TrackingNode', ['particle', 'process', 'volume', 'energy'])

class MyTrackingAction(g4.G4UserTrackingAction):
    """My Tracking Action

    In Geant4, tracks aren't stored throughout the duration of an
    event.  Geant4 deletes a track once its finish tracking all of its
    descendent tracks.

    We would like to access a minimal representation of the tracking
    tree during event analysis.  To do this, we build and store our
    own tree on an event-by-event basis.  For the lack of a better
    alternative, the tree is stored in the global context.
    """

    def PreUserTrackingAction(self, track):
        creator = track.GetCreatorProcess()
        if creator is None:
            process = 'primary'
        else:
            process = creator.GetProcessName()

        global tracking_tree
        if track.GetTrackID() == 1:
            tracking_tree = Tree()
            parent_node = None
        else:
            parent_node = track.GetParentID()

        if not track.GetTrackID() in tracking_tree:
            tracking_tree.create_node(
                identifier=track.GetTrackID(),
                parent=parent_node,
                data=TrackingNode(
                    track.GetDefinition().GetParticleName(),
                    process,
                    track.GetVolume().GetName(),
                    track.GetKineticEnergy()))


class MySteppingAction(g4.G4UserSteppingAction):
    "My Stepping Action"

    def UserSteppingAction(self, step):
        pass

def G4ThreeVector_to_list(x):
    return [x.getX(), x.getY(), x.getZ()]

class SimpleDepositionSD(g4.G4VSensitiveDetector):
    def __init__(self, name, filename):
        g4.G4VSensitiveDetector.__init__(self, name)
        self.filename = filename
        self.position = []
        self.edep = []

    def ProcessHits(self, step, history):
        self.position.append(
            G4ThreeVector_to_list(step.GetPreStepPoint().GetPosition()))
        self.edep.append(step.GetTotalEnergyDeposit())

    def finalize(self, num_events):
        path = os.path.dirname(self.filename)
        if path != '':
            os.makedirs(path, exist_ok=True)
        f = h5py.File(self.filename, 'w')
        f['position'] = np.array(self.position)/mm
        f['edep'] = np.array(self.edep)/keV
        f['edep'].attrs.create('num_events', num_events)
        f.close()

TreeFilter = namedtuple(
    'TreeFilter', ['mode', 'process', 'volume', 'level'])

class BinnedDepositionSD(g4.G4VSensitiveDetector):
    def __init__(self, name, conf):
        g4.G4VSensitiveDetector.__init__(self, name)
        self.filename = conf['File']
        if 'TreeFilter' in conf:
            c = conf['TreeFilter']
            self.tree_filter = TreeFilter(*c[0:3], int(c[3]))
        else:
            self.tree_filter = None

        if 'Group' in conf:
            self.groupname = conf['Group']
        else:
            self.groupname = None
        self.M = compton.build_transformation(conf['Transformation'], mm, deg)
        aeval = asteval.Interpreter(use_numpy=True)
        for q in g4.hepunit.__dict__:
            aeval.symtable[q] = g4.hepunit.__dict__[q]
        self.bin_edges = [aeval(q) for q in conf['BinEdges']]
        self.hist = np.zeros([len(q)-1 for q in self.bin_edges])
        self.update_interval = self.hist.size
        self.position = []
        self.edep = []
        try:
            os.unlink(self.filename)
        except OSError as e:
            pass

    def ProcessHits(self, step, history):
        global tracking_tree

        track = g4.gTrackingManager.GetTrack()
        A = list(tracking_tree.rsearch(track.GetTrackID()))
        A.reverse()

        step = track.GetStep()
        point = step.GetPostStepPoint()
        prestep = step.GetPreStepPoint()

        # if len(A) == 1:
        #     tracking_node = tracking_tree[A[-1]].data
        # else:
        #     tracking_node = tracking_tree[A[-2]].data

        # print(
        #     'SD: {} initial:({},{},{},{:.3g}) {} event={} parent={} track={} step={} process={} KE={:.3f} dE={:.3f}'.format(
        #         A,
        #         tracking_node.particle,
        #         tracking_node.process,
        #         tracking_node.volume,
        #         tracking_node.energy/MeV,
        #         track.GetDefinition().GetParticleName(),
        #         g4.gEventManager.GetNonconstCurrentEvent().GetEventID(),
        #         track.GetParentID(),
        #         track.GetTrackID(),
        #         track.GetCurrentStepNumber(),
        #         point.GetProcessDefinedStep().GetProcessName(),
        #         prestep.GetKineticEnergy()/keV,
        #         step.GetTotalEnergyDeposit()/keV))

        if self.tree_filter is not None:
            mode, process, volume, level = self.tree_filter
            if level >= len(A):
                match = False
            else:
                data = tracking_tree[A[level]].data
                match = (
                    (data.process == process) and
                    (data.volume == volume))
            if mode == 'Exclude' and match == True:
                return
            elif mode == 'Include' and match == False:
                return

        self.position.append(
            G4ThreeVector_to_list(step.GetPreStepPoint().GetPosition()))
        self.edep.append(step.GetTotalEnergyDeposit())
        if len(self.edep) > self.update_interval:
            self.update_histo()
        return

    def update_histo(self):
        if len(self.position)>0:
            M_position = compton.transform(self.M, np.array(self.position))
            B, _ = np.histogramdd(
                M_position, self.bin_edges, weights=np.array(self.edep))
            self.hist += B
            self.position = []
            self.edep = []

    def finalize(self, num_events):
        self.update_histo()
        path = os.path.dirname(self.filename)
        if path != '':
            os.makedirs(path, exist_ok=True)
        fout = h5py.File(self.filename, 'a')
        if self.groupname is not None:
            gout = fout.create_group(self.groupname)
        else:
            gout = fout
        gout['edep'] = self.hist.astype('float32')/MeV
        gout['edep'].attrs.create('num_events', num_events)
        gout['edep'].attrs.create('unit', np.string_('MeV'))
        for i, dset_name in enumerate(['xbin', 'ybin', 'zbin']):
            gout[dset_name] = self.bin_edges[i]/mm
            gout[dset_name].attrs.create('unit', np.string_('mm'))
        fout.close()

TransmissionResult = namedtuple(
    'TransmissionResult', ['position', 'direction', 'energy', 'time'])

class TransmissionSD(g4.G4VSensitiveDetector):
    def __init__(self, name, filename, particles):
        g4.G4VSensitiveDetector.__init__(self, name)
        self.filename = filename
        self.particles = particles
        self.results = {
            p:TransmissionResult([], [], [], []) for p in particles }

    def ProcessHits(self, step, history):
        proc = step.GetPostStepPoint().GetProcessDefinedStep()
        if (proc == None) or (proc.GetProcessName() != 'Transportation'):
            return
        particle_name = str(step.GetTrack().GetDefinition().GetParticleName())
        if particle_name in self.particles:
            point = step.GetPostStepPoint()

            if particle_name in self.results:
                result = self.results[particle_name]
                point = step.GetPostStepPoint()
                result.position.append(
                    G4ThreeVector_to_list(point.GetPosition()))
                result.direction.append(
                    G4ThreeVector_to_list(point.GetMomentumDirection()))
                result.energy.append(point.GetKineticEnergy())
                result.time.append(point.GetGlobalTime())

    def finalize(self, num_events):
        path = os.path.dirname(self.filename)
        if path != '':
            os.makedirs(path, exist_ok=True)
        fout = h5py.File(self.filename, 'w')
        for p, result in self.results.items():
            gout = fout.create_group(p)
            gout['position'] = np.array(result.position)/mm
            gout['direction'] = np.array(result.direction)
            gout['energy'] = np.array(result.energy)/MeV
            gout['time'] = np.array(result.time)/ns
        fout['num_events'] = num_events
        fout.close()

class FlagSD(g4.G4VSensitiveDetector):
    def __init__(self, name, conf):
        g4.G4VSensitiveDetector.__init__(self, name)
        self.M = compton.build_transformation(conf['Transformation'], mm, deg)
        aeval = asteval.Interpreter(use_numpy=True)
        for q in g4.hepunit.__dict__:
            aeval.symtable[q] = g4.hepunit.__dict__[q]
        self.vol = np.array((conf['Volume']))*mm
        self.threshold = conf['Threshold']*MeV
        self.limit_count = conf['LimitCount']
        self.num_flagged = 0
        self.curr_event = -1

    def ProcessHits(self, step, history):
        event_id = g4.gEventManager.GetConstCurrentEvent().GetEventID()
        if event_id != self.curr_event:
            self.curr_event = event_id
            self.tally = 0
        x = G4ThreeVector_to_list(step.GetPreStepPoint().GetPosition())
        Mx = compton.transform(self.M, np.array(x))
        if self.tally < self.threshold:
            if compton.in_volume(self.vol, np.array((Mx,)))[0]:
                self.tally += step.GetTotalEnergyDeposit()
                if self.tally >= self.threshold:
                    self.num_flagged += 1
                    if self.num_flagged <= self.limit_count:
                        g4.gApplyUICommand('/event/keepCurrentEvent')

    def finalize(self, num_events):
        g4.gApplyUICommand('/vis/enable')
        g4.gApplyUICommand('/vis/viewer/flush')

def depth_first_tree_traversal(node):
    q = deque()
    for k, v in node.items():
        if type(v) is dict:
            # set 'Name' field if not already set
            if 'Name' not in v:
                if 'Name' not in node:
                    v['Name'] = k
                else:
                    v['Name'] = node['Name'] + '.' + k
            q.append(v)
            yield v
    while q:
        yield from depth_first_tree_traversal(q.popleft())

class MyGeometry(g4.G4VUserDetectorConstruction):
    def __init__(self, conf):
        self.conf = conf
        g4.G4VUserDetectorConstruction.__init__(self)

    def Construct(self):
        check_overlap = False
        many = False

        geometry_keys = list(self.conf['Geometry'].keys())
        if len(geometry_keys) != 1:
            raise ValueError('Must define exactly one top-level Geometry')
        world_name = geometry_keys[0]

        global geom_s, geom_l, geom_p
        geom_s = {}
        geom_l = {}
        geom_p = {}

        for geom in depth_first_tree_traversal(self.conf['Geometry']):
            geom_type = geom['Type']
            geom_name = geom['Name']
            parent_name = geom_name.rsplit('.', 1)[0]
            if parent_name in geom_p:
                parent_p = geom_p[parent_name]
            else:
                parent_p = None

            if geom_type == 'G4Box':
                solid = g4.G4Box(
                    geom_name, geom['pX']*mm, geom['pY']*mm, geom['pZ']*mm)
            elif geom_type == 'G4Tubs':
                solid = g4.G4Tubs(
                    geom_name, geom['pRMin']*mm, geom['pRMax']*mm,
                    geom['pDz']*mm, geom['pSPhi']*deg, geom['pDPhi']*deg)
            elif geom_type == 'CadMesh':
                mesh = compton.cadmesh.TessellatedMesh(geom['File'])
                if 'SolidName' in geom:
                    solid = mesh.GetSolid(geom['SolidName'])
                else:
                    solid = mesh.GetSolid(0)
            else:
                raise ValueError(
                    "unimplemented geometry type '{}'".format(geom_type))

            logical = g4.G4LogicalVolume(
                solid, g4.gNistManager.FindOrBuildMaterial(
                    geom['Material']), geom_name)
            transform = g4.G4Transform3D()
            if 'Transformation' in geom:
                for operation, value in zip(*geom['Transformation']):
                    translation = g4.G4ThreeVector()
                    rotation = g4.G4RotationMatrix()
                    if operation == 'TranslateX':
                        translation += g4.G4ThreeVector(value*mm, 0, 0)
                    elif operation == 'TranslateY':
                        translation += g4.G4ThreeVector(0, value*mm, 0)
                    elif operation == 'TranslateZ':
                        translation += g4.G4ThreeVector(0, 0, value*mm)
                    elif operation == 'RotateX':
                        rotation.rotateX(value*deg)
                    elif operation == 'RotateY':
                        rotation.rotateY(value*deg)
                    elif operation == 'RotateZ':
                        rotation.rotateZ(value*deg)
                    else:
                        assert(False)
                    transform = (
                        g4.G4Transform3D(rotation, translation)*transform)
            if 'Rotation' in geom:
                euler = np.array(geom['Rotation'])*deg
                rotation = g4.G4RotationMatrix()
                rotation.rotateZ(euler[0])
                rotation.rotateY(euler[1])
                rotation.rotateZ(euler[2])
            else:
                rotation = g4.G4RotationMatrix()
            if 'Translation' in geom:
                translation = g4.G4ThreeVector(
                    *np.array(geom['Translation'])*mm)
            else:
                translation = g4.G4ThreeVector()
            physical = g4.G4PVPlacement(
                g4.G4Transform3D(rotation, translation) * transform,
                geom_name, logical, parent_p, many, 0, check_overlap)

            if 'Visible' in geom:
                logical.SetVisAttributes(g4.G4VisAttributes(geom['Visible']))
            if 'Color' in geom:
                logical.SetVisAttributes(
                    g4.G4VisAttributes(g4.G4Color(*geom['Color'])))

            geom_s[geom_name] = solid
            geom_l[geom_name] = logical
            geom_p[geom_name] = physical

        return geom_p[world_name]

    def ConstructSDandField(self):
        sys.exit()
        pass

def create_fields(conf):
    result = {}

    if 'Fields' not in conf:
        return result

    for name in (conf['Fields']):
        c = conf['Fields'][name]
        field_type = c['Type']
        if field_type == 'ImportedMagneticField':
            field = compton.ImportedMagneticField(c['File'])
            field_manager = g4.gTransportationManager.GetFieldManager()
            field_manager.SetDetectorField(field)
            field_manager.CreateChordFinder(field)
            chord_finder = field_manager.GetChordFinder()
            # field_manager.SetMaximumEpsilonStep(0.001)
            # field_manager.SetMinimumEpsilonStep(5e-5)
            # field_manager.SetDeltaOneStep(0.01*mm)
            # field_manager.SetDeltaIntersection(0.001*mm)
            if 'DeltaChord' in c:
                chord_finder.SetDeltaChord(c['DeltaChord']*mm)
            if 'ScalingFactor' in c:
                field.setScalingFactor(c['ScalingFactor'])
        else:
            raise ValueError(
                "unimplemented Field type '{}'".format(field_type))
        result[name] = field
    return result

def create_materials(conf):
    result = {}

    if 'Materials' not in conf:
        return result

    for name in (conf['Materials']):
        c = conf['Materials'][name]
        mat = g4.gNistManager.FindOrBuildMaterial(name)
        if mat is None:
            atoms = c['AtomicComposition']
            mat = g4.G4Material(name, c['Density']*gram/cm**3, len(atoms[0]))
            for atom_name, num_atoms in zip(*atoms):
                mat.AddElement(
                    g4.gNistManager.FindOrBuildElement(atom_name),
                    num_atoms)
        for property_name in c['Properties']:
            property_table = mat.GetMaterialPropertiesTable()
            if property_table is None:
                property_table = compton.G4MaterialPropertiesTable()
                mat.SetMaterialPropertiesTable(property_table)
            property_conf = c['Properties'][property_name]
            if 'Value' in property_conf:
                property_table.AddConstProperty(
                    property_name, property_conf['Value'])
            else:
                values = np.array(property_conf['Values'])
                if 'PhotonEnergies' in property_conf:
                    photon_energies = np.array(
                        property_conf['PhotonEnergies']) * eV
                else:
                    photon_wavelengths = np.array(
                        property_conf['PhotonWavelengths']) * nanometer
                    photon_energies = h_Planck*c_light/photon_wavelengths
                assert(len(values) == len(photon_energies))
                idx = photon_energies.argsort()
                photon_energies = photon_energies[idx]
                values = values[idx]
                property_table.AddProperty(
                    property_name, photon_energies, values)

        result[name] = mat
    return result


def create_detectors(conf):
    global geom_l
    result = {}

    if 'Detectors' not in conf:
        return result

    global multi_sd
    multi_sd = {}

    for name in (conf['Detectors']):
        c = conf['Detectors'][name]
        sd_type = c['Type']
        if sd_type == 'SimpleDepositionSD':
            sd = SimpleDepositionSD('pbpl/' + name, c['File'])
        elif sd_type == 'BinnedDepositionSD':
            sd = BinnedDepositionSD('pbpl/' + name, c)
        elif sd_type == 'TransmissionSD':
            sd = TransmissionSD('pbpl/' + name, c['File'], c['Particles'])
        elif sd_type == 'FlagSD':
            sd = FlagSD('pbpl/' + name, c)
        else:
            raise ValueError(
                "unimplemented Detector type '{}'".format(sd_type))
        for volume in c['Volumes']:
            if volume not in multi_sd:
                msd = compton.G4MultiSensitiveDetector('pbpl/' + volume)
                geom_l[volume].SetSensitiveDetector(msd)
                multi_sd[volume] = msd
            multi_sd[volume].AddSD(sd)
        result[name] = sd
    return result

def create_event_actions(conf):
    result = {}

    if 'EventActions' not in conf:
        return result

    for name in (conf['EventActions']):
        c = conf['EventActions'][name]
        action_type = c['Type']
        if action_type == 'Status':
            num_events = conf['PrimaryGenerator']['NumEvents']
            update_period = c['UpdatePeriod'] if 'UpdatePeriod' in c else 1.0
            action = StatusEventAction(num_events, update_period)
        else:
            raise ValueError(
                "unimplemented EventAction type '{}'".format(action_type))
        result[name] = action
    return result

def run_main(args):

    global random_engine
    random_engine = g4.RanecuEngine()
    g4.HepRandom.setTheEngine(random_engine)
    g4.HepRandom.setTheSeed(random.randint(0,1e9))

    global detector
    detector = MyGeometry(args.conf)
    g4.gRunManager.SetUserInitialization(detector)

    global physics_list
    physics_list = compton.PhysicsList()
    g4.gRunManager.SetUserInitialization(physics_list)

    global pga
    pga = PrimaryGeneratorAction(args.conf)
    g4.gRunManager.SetUserAction(pga)

    num_events = args.conf['PrimaryGenerator']['NumEvents']

    global t1, t3
    t1 = MyRunAction()
    t3 = MySteppingAction()
    g4.gRunManager.SetUserAction(t1)
    g4.gRunManager.SetUserAction(t3)

    global event_actions
    event_actions = create_event_actions(args.conf)
    for action in event_actions.values():
        g4.gRunManager.SetUserAction(action)

    # global stacking_action
    # stack_action = MyStackingAction()
    # g4.gRunManager.SetUserAction(stack_action)

    global tracking_action
    track_action = MyTrackingAction()
    g4.gRunManager.SetUserAction(track_action)

    global fields
    fields = create_fields(args.conf)

    g4.gRunManager.Initialize()

    global materials
    materials = create_materials(args.conf)
    # for mat in materials.values():
    #     property_table = mat.GetMaterialPropertiesTable()
    #     if property_table is not None:
    #         property_table.DumpTable()

    global detectors
    detectors = create_detectors(args.conf)

    for x in args.macro_filenames:
        g4.gControlExecute(x)

    # g4.gApplyUICommand('/tracking/storeTrajectory 1')
    g4.gRunManager.BeamOn(num_events)

    for k, sd in detectors.items():
        sd.finalize(num_events)

    return 0

def main():
    args = get_args()
    run_main(args)

if __name__ == '__main__':
    sys.exit(main())
