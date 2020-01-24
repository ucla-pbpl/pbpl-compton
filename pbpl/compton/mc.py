# -*- coding: utf-8 -*-
import os, sys, random
import argparse
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

    def Initialize(self, hit_collection):
        pass

    def ProcessHits(self, step, history):
        self.position.append(
            G4ThreeVector_to_list(step.GetPreStepPoint().GetPosition()))
        self.edep.append(step.GetTotalEnergyDeposit())

    def EndOfEvent(self, hit_collection):
        pass

    def finalize(self, num_events):
        path = os.path.split(self.filename)[0]
        if path != '':
            os.makedirs(path, exist_ok=True)
        f = h5py.File(self.filename, 'w')
        f['position'] = np.array(self.position)/mm
        f['edep'] = np.array(self.edep)/keV
        f['edep'].attrs.create('num_events', num_events)
        f.close()

TransmissionResult = namedtuple(
    'TransmissionResult', ['position', 'direction', 'energy'])

class TransmissionSD(g4.G4VSensitiveDetector):
    def __init__(self, name, filename, particles):
        g4.G4VSensitiveDetector.__init__(self, name)
        self.filename = filename
        self.particles = particles
        self.results = { p:TransmissionResult([], [], []) for p in particles }

    def Initialize(self, hit_collection):
        pass

    def ProcessHits(self, step, history):
        proc = step.GetPostStepPoint().GetProcessDefinedStep()
        if (proc == None) or (proc.GetProcessName() != 'Transportation'):
            return
        particle_name = str(step.GetTrack().GetDefinition().GetParticleName())
        if particle_name in self.particles:
            result = self.results[particle_name]
            point = step.GetPostStepPoint()
            result.position.append(
                G4ThreeVector_to_list(point.GetPosition()))
            result.direction.append(
                G4ThreeVector_to_list(point.GetMomentumDirection()))
            result.energy.append(point.GetKineticEnergy())

    def EndOfEvent(self, hit_collection):
        pass

    def finalize(self, num_events):
        path = os.path.split(self.filename)[0]
        if path != '':
            os.makedirs(path, exist_ok=True)
        fout = h5py.File(self.filename, 'w')
        for p, result in self.results.items():
            gout = fout.create_group(p)
            gout['position'] = np.array(result.position)/mm
            gout['direction'] = np.array(result.direction)
            gout['energy'] = np.array(result.energy)/MeV
        fout['num_events'] = num_events
        fout.close()


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

            if 'Color' in geom:
                logical.SetVisAttributes(
                    g4.G4VisAttributes(g4.G4Color(*geom['Color'])))
            if 'Visible' in geom:
                logical.SetVisAttributes(g4.G4VisAttributes(geom['Visible']))

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
            if 'DeltaChord' in c:
                chord_finder.SetDeltaChord(c['DeltaChord']*mm)
            if 'ScalingFactor' in c:
                field.setScalingFactor(c['ScalingFactor'])
        else:
            raise ValueError(
                "unimplemented Field type '{}'".format(field_type))
        result[name] = field
    return result

def create_detectors(conf):
    global geom_l
    result = {}

    if 'Detectors' not in conf:
        return result

    for name in (conf['Detectors']):
        c = conf['Detectors'][name]
        sd_type = c['Type']
        if sd_type == 'SimpleDepositionSD':
            sd = SimpleDepositionSD('pbpl/' + name, c['File'])
        elif sd_type == 'TransmissionSD':
            sd = TransmissionSD('pbpl/' + name, c['File'], c['Particles'])
        else:
            raise ValueError(
                "unimplemented Detector type '{}'".format(sd_type))
        for volume in c['Volumes']:
            geom_l[volume].SetSensitiveDetector(sd)
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

def main():
    args = get_args()

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

    global fields
    fields = create_fields(args.conf)

    g4.gRunManager.Initialize()

    global detectors
    detectors = create_detectors(args.conf)

    for x in args.macro_filenames:
        g4.gControlExecute(x)

    g4.gRunManager.BeamOn(num_events)

    for k, sd in detectors.items():
        sd.finalize(num_events)

    return 0

if __name__ == '__main__':
    sys.exit(main())
