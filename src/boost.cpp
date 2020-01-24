// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <Python.h>

#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <stdexcept>
#include <memory>

#include <boost/preprocessor/seq/for_each.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/python/call.hpp>
#include <boost/python/list.hpp>
//#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

#include "PhysicsList.h"
#include <G4ParticleGun.hh>
#include <G4UIExecutive.hh>
#include "ImportedMagneticField.h"
#include <CADMesh.hh>
#include <G4VSolid.hh>
#include <G4SDManager.hh>
#include <boost/multi_array.hpp>

#include <G4AssemblyVolume.hh>

namespace bp = boost::python;

template<class T> struct VecToList
{
    static PyObject* convert(const std::vector<T>& vec) {
	boost::python::list* l = new boost::python::list();
	for(size_t i = 0; i < vec.size(); i++)
	    (*l).append(vec[i]);
        return l->ptr();
    }
};

typedef std::map<CADMesh::File::Type, G4String> map_t;

void export_cadmesh()
{
    bp::object fileModule(
        bp::handle<>(bp::borrowed(
                         PyImport_AddModule("compton.cadmesh"))));
    bp::scope().attr("cadmesh") = fileModule;
    bp::scope file_scope = fileModule;

//    G4VSolid* (CADMesh::*bah)(int) = &CADMesh::TessellatedMesh;
    bp::class_<CADMesh::TessellatedMesh>(
        "TessellatedMesh", bp::init<char *>())
        .def("IsValidForNavigation", &CADMesh::TessellatedMesh::IsValidForNavigation)
        .def("GetFileName", &CADMesh::TessellatedMesh::GetFileName)
        .def("GetFileType", &CADMesh::TessellatedMesh::GetFileType)
        .def("GetVerbose", &CADMesh::TessellatedMesh::GetVerbose)
        .def("SetVerbose", &CADMesh::TessellatedMesh::SetVerbose)
        .def("GetScale", &CADMesh::TessellatedMesh::GetScale)
        .def("SetScale", &CADMesh::TessellatedMesh::SetScale)
        .def("GetOffset", &CADMesh::TessellatedMesh::GetOffset)
        .def("SetOffset", &CADMesh::TessellatedMesh::SetOffset)
        .def("GetAssembly", &CADMesh::TessellatedMesh::GetAssembly,
             bp::return_internal_reference<>())
        .def<G4VSolid* (CADMesh::TessellatedMesh::*)(G4String)>(
            "GetSolid", &CADMesh::TessellatedMesh::GetSolid,
            bp::return_internal_reference<>())
        .def<G4VSolid* (CADMesh::TessellatedMesh::*)(G4int)>(
            "GetSolid", &CADMesh::TessellatedMesh::GetSolid,
            bp::return_internal_reference<>());


    // bp::object fileModule(
    //     bp::handle<>(bp::borrowed(
    //                      PyImport_AddModule("compton.cadmesh"))));
    // bp::scope().attr("file") = fileModule;
    // bp::scope file_scope = fileModule;

    bp::enum_<CADMesh::File::Type>(
        "Type",
R"(


=========  =========  ==============================
Attribute  Extension  TypeName
=========  =========  ==============================
Unknown    unknown    Unknown File Format
PLY        ply        Stanford Triangle Format (PLY)
STL        stl        Stereolithography (STL)
DAE        dae        COLLADA (DAE)
OBJ        obj        Wavefront (OBJ)
TET        tet        TetGet (TET)
OFF        off        Object File Format (OFF)
=========  =========  ==============================

)")
        .value("PLY", CADMesh::File::Type::PLY)
        .value("STL", CADMesh::File::Type::STL)
        .value("DAE", CADMesh::File::Type::DAE)
        .value("OBJ", CADMesh::File::Type::OBJ)
        .value("TET", CADMesh::File::Type::TET)
        .value("OFF", CADMesh::File::Type::OFF);


    bp::class_<map_t>("TypeMap")
        .def(bp::map_indexing_suite<map_t>());
    bp::scope().attr("Extension") = CADMesh::File::Extension;
    bp::scope().attr("TypeString") = CADMesh::File::TypeString;
    bp::scope().attr("TypeName") = CADMesh::File::TypeName;
}


BOOST_PYTHON_MODULE(boost)
{
    Py_Initialize();

    bp::docstring_options local_docstring_options(true, false, false);

    bp::to_python_converter<
	std::vector<std::string, class std::allocator<std::string> >,
	VecToList<std::string> >();

    bp::to_python_converter<
        std::vector<float, class std::allocator<float> >,
        VecToList<float> >();

    bp::class_<G4AssemblyVolume, G4AssemblyVolume*, boost::noncopyable>
        ("G4AssemblyVolume", "assembly class", bp::no_init);

    bp::class_<G4SDManager, G4SDManager*, boost::noncopyable>
        ("G4SDManager", "G4SDManager class", bp::no_init)
        .def("GetSDMpointer", &G4SDManager::GetSDMpointer,
             bp::return_value_policy<bp::reference_existing_object>())
        .staticmethod("GetSDMpointer")
        .def("AddNewDetector", &G4SDManager::AddNewDetector);

    bp::class_<PhysicsList, boost::shared_ptr<PhysicsList>,
               bp::bases<G4VUserPhysicsList> >
        ("PhysicsList", "PBPL physics list");

    bp::class_<
        ImportedMagneticField, ImportedMagneticField*,
        bp::bases<G4Field, G4MagneticField> >
        ("ImportedMagneticField",
R"(Represent Geant4 magnetic field using imported data.

* Field is loaded from a pbpl-compton HDF5 file.
* Field data is stored on a uniform Cartesian grid.
* Field values are linearly interpolated.

.. code-block:: ipython

  In [1]: from pbpl import compton
  In [2]: from pbpl.units import *
  In [3]: import numpy as np
  In [4]: B = compton.ImportedMagneticField('bfield.h5')
  In [5]: loc = np.vector((1,2,3))*mm
  In [6]: print('B({}) = {} T'.format(B.GetFieldValue(loc))/tesla)

)")
        .def(bp::init<const std::string&>())
        .def("eval", &ImportedMagneticField::eval)
        .def("dumpInfo", &ImportedMagneticField::dumpInfo)
        .def("loadField", &ImportedMagneticField::loadField,
R"(loadField(filename)

Load HDF5 file.  File contains following mandatory datasets:
  * xvals[Nx] (unit=meter)
  * yvals[Nx] (unit=meter)
  * zvals[Nx] (unit=meter)
  * B_field[3, Nx, Ny, Nz] (unit=tesla)

Args:
  filename (str): HDF5 filename
)")
        .def("setScalingFactor", &ImportedMagneticField::setScalingFactor,
R"(setScalingFactor(value)

Args:
  value (float): set scaling factor (initially defaults to 1.0)
)");

    export_cadmesh();
//    export_cadmesh_file();
}
