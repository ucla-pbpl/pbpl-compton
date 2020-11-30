// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include <iostream>
#include "PhysicsList.h"
#include "Particles.h"
#include "PhysicsListEMstd.h"
#include <G4SystemOfUnits.hh>
#include <G4EmStandardPhysics_option4.hh>
#include <G4OpticalPhysics.hh>

PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
    // default cut value  (1.0mm)
    defaultCutValue = 1.*mm;
    SetVerboseLevel(1);

    // particles
    RegisterPhysics(new Particles);

    // EM Physics
//   RegisterPhysics(new G4EmStandardPhysics_option4);
    RegisterPhysics(new PhysicsListEMstd);
    G4OpticalPhysics* op = new G4OpticalPhysics;
    RegisterPhysics(op);
}


PhysicsList::~PhysicsList()
{
}


void PhysicsList::SetCuts()
{
    // G4VUserPhysicsList::SetCutsWithDefault() method sets
    // the default cut value for all particle types
    SetCutsWithDefault();
}
