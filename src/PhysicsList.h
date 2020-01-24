// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#ifndef PHYSICS_LIST_H
#define PHYSICS_LIST_H

#include <G4VModularPhysicsList.hh>
#include <globals.hh>

class PhysicsList: public G4VModularPhysicsList
{
public:
    PhysicsList();
    ~PhysicsList();
    virtual void SetCuts();
};

#endif
