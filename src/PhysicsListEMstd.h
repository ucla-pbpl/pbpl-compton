// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#ifndef PHYSICS_LIST_EM_STD_H
#define PHYSICS_LIST_EM_STD_H

#include <G4VPhysicsConstructor.hh>

class PhysicsListEMstd : public G4VPhysicsConstructor
{
public:
    PhysicsListEMstd();
    ~PhysicsListEMstd();
    virtual void ConstructParticle();
    virtual void ConstructProcess();
};

#endif
