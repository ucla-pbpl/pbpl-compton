// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include <G4MaterialPropertiesTable.hh>
#include "Exception.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace pyG4MaterialPropertiesTable {

void AddConstProperty(
    G4MaterialPropertiesTable* p, const std::string& key, double value)
{
    p->AddConstProperty(key.c_str(), value);
}

void AddProperty(
    G4MaterialPropertiesTable* p, const std::string& key,
    const np::ndarray& energy, const np::ndarray& value)
{
    if (energy.get_nd() != 1)
        pbpl_throw("energy array must be 1D");
    if (value.get_nd() != 1)
        pbpl_throw("value array must be 1D");
    if (energy.shape(0) != value.shape(0))
        pbpl_throw("energy and value arrays must have same length");

    auto energy_d = energy.view(np::dtype::get_builtin<double>());
    auto value_d = value.view(np::dtype::get_builtin<double>());
    p->AddProperty(
        key.c_str(),
        (double *) energy_d.get_data(),
        (double *) value_d.get_data(),
        energy_d.shape(0));
}

}

void export_G4MaterialPropertiesTable()
{
    std::cout << "export_G4MaterialPropertiesTable\n";
    bp::class_<G4MaterialPropertiesTable, G4MaterialPropertiesTable*,
               boost::noncopyable>
        ("G4MaterialPropertiesTable", "material properties table class")
        .def(bp::init<>())
        .def("AddConstProperty",
             &pyG4MaterialPropertiesTable::AddConstProperty)
        .def("AddProperty", &pyG4MaterialPropertiesTable::AddProperty)
        .def("DumpTable", &G4MaterialPropertiesTable::DumpTable);
}
