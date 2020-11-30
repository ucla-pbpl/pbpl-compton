// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "PhysicsList.h"
#include "ImportedMagneticField.h"
#include <G4VSolid.hh>
#include <G4SDManager.hh>
#include <G4AssemblyVolume.hh>

#include "Exception.h"
#include "pyCADMesh.h"
#include "pyG4MultiSensitiveDetector.h"
#include "pyG4MaterialPropertiesTable.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

template<class T> struct VecToList
{
    static PyObject* convert(const std::vector<T>& vec) {
	boost::python::list* l = new boost::python::list();
	for(size_t i = 0; i < vec.size(); i++)
	    (*l).append(vec[i]);
        return l->ptr();
    }
};

BOOST_PYTHON_MODULE(boost)
{
    Py_Initialize();

    bp::docstring_options local_docstring_options(true, false, false);

    np::initialize();

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

    export_CADMesh();
    export_G4MultiSensitiveDetector();
    export_G4MaterialPropertiesTable();
}
