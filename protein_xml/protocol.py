
from simtk.openmm.app import * 
from simtk.openmm import * 
from simtk.unit import * 
from sys import stdout 
import parmed as pmd
from parmed.openmm import XmlFile

pdbfile = PDBFile('QUBE_pro.pdb')
ff = ForceField("QUBE_pro.xml")
system = ff.createSystem( pdbfile.topology)
integrator = openmm.VerletIntegrator(1.0 )
context = openmm.Context(system, integrator)
context.setPositions(pdbfile.positions)
# Create parmed Structure
structure = parmed.openmm.topsystem.load_topology( pdbfile.topology, system, pdbfile.positions)
# Write AMBER parameter/crd
structure.save('system.prmtop', overwrite=True)
structure.save('system.crd', format='rst7', overwrite=True)
