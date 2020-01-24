// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#ifndef PARTICLES_H
#define PARTICLES_H

#include <G4VPhysicsConstructor.hh>

class Particles : public G4VPhysicsConstructor
{
public:
    Particles();
    ~Particles();
    virtual void ConstructParticle();
    virtual void ConstructProcess();
};

#endif
