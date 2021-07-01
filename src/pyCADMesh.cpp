// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "Exception.h"
#include <CADMesh.hh>

namespace bp = boost::python;

typedef std::map<CADMesh::File::Type, G4String> map_t;

void export_CADMesh()
{
    bp::object fileModule(
        bp::handle<>(bp::borrowed(
                         PyImport_AddModule("compton.cadmesh"))));
    bp::scope().attr("cadmesh") = fileModule;
    bp::scope file_scope = fileModule;

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
