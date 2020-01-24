// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <G4SystemOfUnits.hh>
#include <G4Exp.hh>
#include <G4AutoLock.hh>
#include "ImportedMagneticField.h"

#undef H5_USE_BOOST
#define H5_USE_BOOST
#include "highfive/H5File.hpp"
#include "highfive/H5Easy.hpp"

namespace
{
    G4Mutex ImportedMagneticFieldMutex = G4MUTEX_INITIALIZER;
}

void ImportedMagneticField::loadField(const std::string& filename)
{
    G4AutoLock lock(&ImportedMagneticFieldMutex);

    HighFive::File fin(filename, HighFive::File::ReadOnly);
    const char *dset_names[] = { "/xvals", "/yvals", "/zvals" };
    for (unsigned i=0; i<3; ++i) {
        xvals[i] = H5Easy::load<std::vector<float> >(fin, dset_names[i]);
        for (float &x : xvals[i]) {
            x *= meter;
        }
        x0[i] = xvals[i].front();
        x1[i] = xvals[i].back();
        N[i] = xvals[i].size();
        dx[i] = (x1[i] - x0[i]) / (N[i] - 1);
    }

    HighFive::DataSet dset = fin.getDataSet("/B_field");
    field.resize(dset.getSpace().getDimensions());
    dset.read(field);
    for (unsigned i=0; i<field.num_elements(); ++i) {
        field.data()[i] *= tesla;
    }
    scaling_factor = 1.0;
}


std::vector<double> ImportedMagneticField::eval(
    double x, double y, double z) const
{
    double point[4] = { x, y, z, 0.0 };
    double result[6];
    GetFieldValue(point, result);
    return std::vector<double> { result[0], result[1], result[2] };
}


std::string ImportedMagneticField::dumpInfo() const
{
    std::ostringstream oss;
    const auto shape = field.shape();
    for (unsigned i=0; i<3; ++i) {
        oss << "index=" << i << ": "
            << shape[1+i] << ' '
            << x0[i] << ' '
            << x1[i] << '\n';
    }
    return oss.str();
}

void ImportedMagneticField::GetFieldValue(const double point[4], double *result) const
{
    result[0] = 0.0;
    result[1] = 0.0;
    result[2] = 0.0;
    result[3] = 0.0;
    result[4] = 0.0;
    result[5] = 0.0;
    double x[3] = { point[0], point[1], point[2] };
    //double t = point[3];

    if (x[0]>=x0[0] && x[0]<=x1[0] &&
        x[1]>=x0[1] && x[1]<=x1[1] &&
        x[2]>=x0[2] && x[2]<=x1[2]) {

        int idx[3];
        double xd[3];
        for (unsigned i=0; i<3; ++i) {
            double temp;
            xd[i] = std::modf((x[i]-x0[i])/dx[i], &temp);
            idx[i] = static_cast<int>(std::floor(temp));
        }

        // TODO: Use cached cXXXX values if idx[i] values unchanged
        //       from previous evaluation.  Make sure optimization
        //       actually performs better.

        for (unsigned i=0; i<3; ++i) {
            const float c000 = field[i][idx[0]  ][idx[1]  ][idx[2]  ];
            const float c001 = field[i][idx[0]  ][idx[1]  ][idx[2]+1];
            const float c010 = field[i][idx[0]  ][idx[1]+1][idx[2]  ];
            const float c011 = field[i][idx[0]  ][idx[1]+1][idx[2]+1];
            const float c100 = field[i][idx[0]+1][idx[1]  ][idx[2]  ];
            const float c101 = field[i][idx[0]+1][idx[1]  ][idx[2]+1];
            const float c110 = field[i][idx[0]+1][idx[1]+1][idx[2]  ];
            const float c111 = field[i][idx[0]+1][idx[1]+1][idx[2]+1];

            result[i] = (
                c000*(1-xd[0])*(1-xd[1])*(1-xd[2]) +
                c001*(1-xd[0])*(1-xd[1])*xd[2]  +
                c010*(1-xd[0])*xd[1]*(1-xd[2]) +
                c011*(1-xd[0])*xd[1]*xd[2]  +
                c100*xd[0]*(1-xd[1])*(1-xd[2]) +
                c101*xd[0]*(1-xd[1])*xd[2] +
                c110*xd[0]*xd[1]*(1-xd[2]) +
                c111*xd[0]*xd[1]*xd[2]);
        }
    }
    for (unsigned i=0; i<6; ++i)
        result[i] *= scaling_factor;
}
