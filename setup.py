#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension, find_packages
import sys
import sysconfig
import subprocess
import shlex
import re as regex

with open('README.rst') as f:
    readme = f.read()

with open('HISTORY.rst') as f:
    history = f.read()

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

geant4_cflags = subprocess.run(
    ['geant4-config', '--cflags'], capture_output=True).stdout.decode('utf-8')
pattern = regex.compile('-I(.+)')
geant4_include_dirs = [
    pattern.match(x)[1] for x in shlex.split(geant4_cflags)
    if pattern.match(x) is not None]

geant4_libflags = subprocess.run(
    ['geant4-config', '--libs'], capture_output=True).stdout.decode('utf-8')
pattern = regex.compile('-L(.+)')
geant4_lib_dirs = [
    pattern.match(x)[1] for x in shlex.split(geant4_libflags)
    if pattern.match(x) is not None]
pattern = regex.compile('-l(.+)')
geant4_libs = [
    pattern.match(x)[1] for x in shlex.split(geant4_libflags)
    if pattern.match(x) is not None]

setup(
    name='pbpl-compton',
    version='0.1.0',
    description='Python package for design and simulation \\\
of FACET-II gamma diagnostics',
    long_description=readme + '\n\n' + history,
    author='Brian Naranjo',
    author_email='brian.naranjo@gmail.com',
    url='https://github.com/bnara/pbpl-compton',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    keywords='UCLA PBPL Compton spectrometer gamma particle tracker',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    ext_modules=[
        Extension(
            'pbpl.compton.boost',
            ['src/boost.cpp',
             'src/ImportedMagneticField.cpp',
             'src/Particles.cpp',
             'src/PhysicsListEMstd.cpp',
             'src/PhysicsList.cpp'],
            include_dirs=[
                *geant4_include_dirs,
                '/usr/include/hdf5/serial',
                '/opt/cadmesh/install/include'],
            library_dirs=[
                *geant4_lib_dirs,
                '/opt/cadmesh/install/lib'],
            libraries=[
                *geant4_libs,
                'boost_python3',
                'cadmesh', 'tet', 'assimp'],
            extra_compile_args=[
                '-DG4INTY_USE_XT -DG4VIS_USE_OPENGL -DG4UI_USE_TCSH -DG4INTY_USE_QT -DG4UI_USE_QT -DG4VIS_USE_OPENGLQT  -DG4VIS_USE_RAYTRACERX -DG4VIS_USE_OPENGLX -W -Wall -pedantic -Wno-non-virtual-dtor -Wno-long-long -Wwrite-strings -Wpointer-arith -Woverloaded-virtual -Wno-variadic-macros -Wshadow -pipe -DG4USE_STD11 -std=c++11'])],
    entry_points = {
        'console_scripts':
        ['pbpl-compton-mc = pbpl.compton.mc:main',
         'pbpl-compton-convert-field = pbpl.compton.convert_field:main',
         'pbpl-compton-extrude-vrml = pbpl.compton.extrude_vrml:main',
         'pbpl-compton-convert-trajectories = pbpl.compton.convert_trajectories:main',
         'pbpl-compton-reduce-edep = pbpl.compton.reduce_edep:main',
         'pbpl-compton-plot-deposition = pbpl.compton.plot_deposition:main',
         'pbpl-compton-build-collimator = pbpl.compton.build_collimator:main',
         'pbpl-compton-calc-map-particles = pbpl.compton.calc_map_particles:main',
        ]
    },
    data_files=[
        ('share/lil-cpt',
         ['share/lil-cpt/lil-cpt.toml',
          'share/lil-cpt/lil_cpt.py',
          'share/lil-cpt/vis.mac']),
        ('share/lil-cpt/cad',
         ['share/lil-cpt/cad/lil-cpt.stl',
          'share/lil-cpt/cad/lil-cpt-magnet.stl']),
        ('share/lil-cpt/field',
         ['share/lil-cpt/field/B-field-2mm.h5',
          'share/lil-cpt/field/B-field-2mm.txt']),
        ('share/lil-cpt/cst',
         ['share/lil-cpt/cst/lil-cpt.mcs',
          'share/lil-cpt/cst/calc-enge.py'])],
    test_suite='tests',
    tests_require=test_requirements,
    namespace_packages=['pbpl']
)
