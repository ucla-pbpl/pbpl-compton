// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include "Exception.h"
#include <G4MultiSensitiveDetector.hh>

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace pyG4MultiSensitiveDetector {

class CB_G4MultiSensitiveDetector :
        public G4MultiSensitiveDetector,
        public bp::wrapper<G4MultiSensitiveDetector> {

public:
    CB_G4MultiSensitiveDetector() : G4MultiSensitiveDetector("") { }
    CB_G4MultiSensitiveDetector(const G4String& name)
        : G4MultiSensitiveDetector(name) { }
    ~CB_G4MultiSensitiveDetector() { }

    // G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) {
    //   return get_override("ProcessHits")(&aStep, &ROhist);
    // }
};

}

using namespace pyG4MultiSensitiveDetector;

void export_G4MultiSensitiveDetector()
{
    bp::class_<CB_G4MultiSensitiveDetector, bp::bases<G4VSensitiveDetector>,
               boost::noncopyable>
        ("G4MultiSensitiveDetector", "base class of senstive detector")
        .def(bp::init<const G4String&>())
    // .def("Initialize",      &G4MultiSensitiveDetector::Initialize)
    // .def("EndOfEvent",      &G4MultiSensitiveDetector::EndOfEvent)
    // .def("clear",           &G4MultiSensitiveDetector::clear)
    // .def("DrawAll",         &G4MultiSensitiveDetector::DrawAll)
    // .def("PrintAll",        &G4MultiSensitiveDetector::PrintAll)
    // .def("Hit",             &G4MultiSensitiveDetector::Hit)
    // .def("ProcessHits", pure_virtual(&CB_G4MultiSensitiveDetector::ProcessHits))
    // // ---
    // .def("SetROgeometry",   &G4MultiSensitiveDetector::SetROgeometry)
    // .def("GetNumberOfCollections",
	//  &G4MultiSensitiveDetector::GetNumberOfCollections)
    // .def("GetCollectionName", &G4MultiSensitiveDetector::GetCollectionName)
    // .def("SetVerboseLevel", &G4MultiSensitiveDetector::SetVerboseLevel)
    // .def("Activate",        &G4MultiSensitiveDetector::Activate)
    // .def("isActive",        &G4MultiSensitiveDetector::isActive)
    // .def("GetName",         &G4MultiSensitiveDetector::GetName)
    // .def("GetPathName",     &G4MultiSensitiveDetector::GetPathName)
    // .def("GetFullPathName", &G4MultiSensitiveDetector::GetFullPathName)
    // .def("GetROgeometry",   &G4MultiSensitiveDetector::GetROgeometry,
    //      bp::return_internal_reference<>())
        .def("AddSD", &G4MultiSensitiveDetector::AddSD);
}
