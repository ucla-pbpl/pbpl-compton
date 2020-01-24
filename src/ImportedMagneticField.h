// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#ifndef IMPORTED_MAGNETIC_FIELD_H
#define IMPORTED_MAGNETIC_FIELD_H

#include <G4MagneticField.hh>

#include <fstream>
#include <vector>
#include <string>
#include <boost/multi_array.hpp>

class ImportedMagneticField : public G4MagneticField
{
public:
    ImportedMagneticField() = default;
    ImportedMagneticField(const std::string& filename) {
        loadField(filename);
    }
    void loadField(const std::string& filename);
    void GetFieldValue(const double Point[4], double *field) const;
    std::vector<double> eval(double x, double y, double z) const;
    std::string dumpInfo() const;
    void setScalingFactor(float val) { scaling_factor = val; }
private:
    typedef boost::multi_array<float, 4> field_type;
    field_type field;
    std::vector<float> xvals[3];
    float x0[3], x1[3], dx[3];
    unsigned N[3];
    float scaling_factor;
};

#endif
