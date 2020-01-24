// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <iostream>
#include "PhysicsList.h"
#include "Particles.h"
#include "PhysicsListEMstd.h"
#include <G4SystemOfUnits.hh>


PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
    // default cut value  (1.0mm)
    defaultCutValue = 1.*mm;
    SetVerboseLevel(1);

    // particles
    RegisterPhysics(new Particles);

    // EM Physics
    RegisterPhysics(new PhysicsListEMstd);
}


PhysicsList::~PhysicsList()
{
}


void PhysicsList::SetCuts()
{
    //  " G4VUserPhysicsList::SetCutsWithDefault" method sets
    //   the default cut value for all particle types
    SetCutsWithDefault();
}
