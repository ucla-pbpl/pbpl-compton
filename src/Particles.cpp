// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#include "Particles.h"

#include <G4LeptonConstructor.hh>
#include <G4BosonConstructor.hh>
#include <G4MesonConstructor.hh>
#include <G4BaryonConstructor.hh>
#include <G4ShortLivedConstructor.hh>
#include <G4IonConstructor.hh>


Particles::Particles() : G4VPhysicsConstructor("Particles")
{
}


Particles::~Particles()
{
}


void Particles::ConstructParticle()
{
    G4LeptonConstructor::ConstructParticle();
    G4BosonConstructor::ConstructParticle();
    G4MesonConstructor::ConstructParticle();
    G4BaryonConstructor::ConstructParticle();
    G4ShortLivedConstructor::ConstructParticle();
    G4IonConstructor::ConstructParticle();
}


void Particles::ConstructProcess()
{
}
