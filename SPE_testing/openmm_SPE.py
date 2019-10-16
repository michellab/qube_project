from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils import constants
from QUBEKit.utils.decorators import timer_logger

import numpy as np

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit

import xml.etree.ElementTree as ET

from copy import deepcopy



pdb = app.PDBFile('MOL.pdb')
forcefield = app.ForceField('MOL.xml')
modeller = app.Modeller(pdb.topology, pdb.positions)  # set the initial positions from the pdb
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

# Check what combination rule we should be using from the xml
xmlstr = open('MOL.xml').read()
# check if we have opls combination rules if the xml is present
try:
    self.combination = ET.fromstring(xmlstr).find('NonbondedForce').attrib['combination']
except AttributeError:
    pass
except KeyError:
    pass

# if self.combination == 'opls':
#     print('OPLS combination rules found in xml file')
#     self.opls_lj()

forces = {system.getForce(index).__class__.__name__: system.getForce(index)
          for index in range(system.getNumForces())}
# Use the nonbonded_force to get the same rules
nonbonded_force = forces['NonbondedForce']
lorentz = mm.CustomNonbondedForce(
    'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')
lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
lorentz.addPerParticleParameter('sigma')
lorentz.addPerParticleParameter('epsilon')
lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
system.addForce(lorentz)

l_j_set = {}
# For each particle, calculate the combination list again
for index in range(nonbonded_force.getNumParticles()):
    charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
    l_j_set[index] = (sigma, epsilon, charge)
    lorentz.addParticle([sigma, epsilon])
    nonbonded_force.setParticleParameters(index, charge, 0, 0)

for i in range(nonbonded_force.getNumExceptions()):
    (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
    # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
    lorentz.addExclusion(p1, p2)
    if eps._value != 0.0:
        charge = 0.5 * (l_j_set[p1][2] * l_j_set[p2][2])
        sig14 = np.sqrt(l_j_set[p1][0] * l_j_set[p2][0])
        nonbonded_force.setExceptionParameters(i, p1, p2, charge, sig14, eps)
   


temperature = 298  * unit.kelvin 
integrator = mm.LangevinIntegrator(temperature, 5 / unit.picoseconds, 0.001 * unit.picoseconds)

simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)


state = simulation.context.getState(getEnergy=True)

energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

print(energy)
