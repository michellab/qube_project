

import os

import zipfile
import os
import re
import sys

from Sire.Base import *

# Make sure that the OPENMM_PLUGIN_DIR enviroment variable is set correctly.
os.environ["OPENMM_PLUGIN_DIR"] = getLibDir() + "/plugins"

from Sire.IO import *
from Sire.Mol import *
from Sire.CAS import *
from Sire.System import *
from Sire.Move import *
from Sire.MM import *
from Sire.FF import *
from Sire.Units import *
from Sire.Vol import *
from Sire.Maths import *
from Sire.Qt import *
from Sire.ID import *
from Sire.Config import *
from Sire.Analysis import *
from Sire.Tools.DCDFile import *
from Sire.Tools import Parameter, resolveParameters
import Sire.Stream
import time
import numpy as np

####################################################################################################
#
#   Config file parameters
#
####################################################################################################
VirtualSite_flag =  Parameter("vSites flag", True, """ Please work""")

gpu = Parameter("gpu", 0, """The device ID of the GPU on which to run the simulation.""")

rf_dielectric = Parameter("reaction field dielectric",0,
                          """Dielectric constant to use if the reaction field cutoff method is used.""")

temperature = Parameter("temperature", 25 * celsius, """Simulation temperature""")

pressure = Parameter("pressure", 1 * atm, """Simulation pressure""")

topfile = Parameter("topfile", "MOL.prm7",
                    """File name of the topology file containing the system to be simulated.""")

crdfile = Parameter("crdfile", "MOL.rst7",
                    """File name of the coordinate file containing the coordinates of the
                       system to be simulated.""")

s3file = Parameter("s3file", "SYSTEM.s3",
                   """Filename for the system state file. The system state after topology and and coordinates
                   were loaded are saved in this file.""")

restart_file = Parameter("restart file", "sim_restart.s3",
                         """Filename of the restart file to use to save progress during the simulation.""")

dcd_root = Parameter("dcd root", "traj", """Root of the filename of the output DCD trajectory files.""")

nmoves = Parameter("nmoves", 1000, """Number of Molecular Dynamics moves to be performed during the simulation.""")

random_seed = Parameter("random seed", None, """Random number seed. Set this if you
                         want to have reproducible simulations.""")

ncycles = Parameter("ncycles", 5,
                    """The number of MD cycles. The total elapsed time will be nmoves*ncycles*timestep""")

maxcycles = Parameter("maxcycles",99999,
                      """The maximum number of MD cycles to carry out. Useful to restart simulations from a checkpoint""")

ncycles_per_snap = Parameter("ncycles_per_snap", 1, """Number of cycles between saving snapshots""")

save_coords = Parameter("save coordinates", True, """Whether or not to save coordinates.""")

buffered_coords_freq = Parameter("buffered coordinates frequency", 1,
                                 """The number of time steps between saving of coordinates during
                                 a cycle of MD. 0 disables buffering.""")
minimal_coordinate_saving = Parameter("minimal coordinate saving", False, "Reduce the number of coordiantes writing for states"
                                                                    "with lambda in ]0,1[")

time_to_skip = Parameter("time to skip", 0 * picosecond, """Time to skip in picoseconds""")

minimise = Parameter("minimise", True, """Whether or not to perform minimization before the simulation.""")

minimise_tol = Parameter("minimise tolerance", 1, """Tolerance used to know when minimization is complete.""")

minimise_max_iter = Parameter("minimise maximum iterations", 10000, """Maximum number of iterations for minimization.""")

equilibrate = Parameter("equilibrate", True , """Whether or not to perform equilibration before dynamics.""")

equil_iterations = Parameter("equilibration iterations", 20000, """Number of equilibration steps to perform.""")

equil_timestep = Parameter("equilibration timestep", 0.5 * femtosecond, """Timestep to use during equilibration.""")

combining_rules = Parameter("combining rules", "geometric",
                            """Combining rules to use for the non-bonded interactions.""")

timestep = Parameter("timestep", 2 * femtosecond, """Timestep for the dynamics simulation.""")

platform = Parameter("platform", "OpenCL", """Which OpenMM platform should be used to perform the dynamics.""")

precision = Parameter("precision", "mixed", """The floating point precision model to use during dynamics.""")

constraint = Parameter("constraint", "hbonds", """The constraint model to use during dynamics.""")

cutoff_type = Parameter("cutoff type", "nocutoff", """The cutoff method to use during the simulation.""")

cutoff_dist = Parameter("cutoff distance", 10 * angstrom,
                        """The cutoff distance to use for the non-bonded interactions.""")

integrator_type = Parameter("integrator", "leapfrogverlet", """The integrator to use for dynamics.""")

inverse_friction = Parameter("inverse friction", 0.1 * picosecond,
                             """Inverse friction time for the Langevin thermostat.""")

andersen = Parameter("thermostat", True,
                     """Whether or not to use the Andersen thermostat (needed for NVT or NPT simulation).""")

barostat = Parameter("barostat", False, """Whether or not to use a barostat (needed for NPT simulation).""")

andersen_frequency = Parameter("andersen frequency", 10.0, """Collision frequency in units of (1/ps)""")

barostat_frequency = Parameter("barostat frequency", 25,
                               """Number of steps before attempting box changes if using the barostat.""")

lj_dispersion = Parameter("lj dispersion", False, """Whether or not to calculate and include the LJ dispersion term.""")

cmm_removal = Parameter("center of mass frequency", 10,
                        "Frequency of which the system center of mass motion is removed.""")

center_solute = Parameter("center solute", False,
                          """Whether or not to centre the centre of geometry of the solute in the box.""")

use_restraints = Parameter("use restraints", False, """Whether or not to use harmonic restaints on the solute atoms.""")

k_restraint = Parameter("restraint force constant", 100.0, """Force constant to use for the harmonic restaints.""")

heavy_mass_restraint = Parameter("heavy mass restraint", 1.10,
                                 """Only restrain solute atoms whose mass is greater than this value.""")

unrestrained_residues = Parameter("unrestrained residues", ["WAT", "HOH"],
                                  """Names of residues that are never restrained.""")

freeze_residues = Parameter("freeze residues", False, """Whether or not to freeze certain residues.""")

frozen_residues = Parameter("frozen residues", ["LGR", "SIT", "NEG", "POS"],
                            """List of residues to freeze if 'freeze residues' is True.""")


use_distance_restraints = Parameter("use distance restraints", False,
                                    """Whether or not to use restraints distances between pairs of atoms.""")

distance_restraints_dict = Parameter("distance restraints dictionary", {},
                                     """Dictionary of pair of atoms whose distance is restrained, and restraint
                                     parameters. Syntax is {(atom0,atom1):(reql, kl, Dl)} where atom0, atom1 are atomic
                                     indices. reql the equilibrium distance. Kl the force constant of the restraint.
                                     D the flat bottom radius. WARNING: PBC distance checks not implemented, avoid
                                     restraining pair of atoms that may diffuse out of the box.""")

hydrogen_mass_repartitioning_factor = Parameter("hydrogen mass repartitioning factor",None,
                                     """If not None and is a number, all hydrogen atoms in the molecule will
                                        have their mass increased by the input factor. The atomic mass of the heavy atom
                                        bonded to the hydrogen is decreased to keep the mass constant.""")

## Free energy specific keywords
morphfile = Parameter("morphfile", "MORPH.pert",
                      """Name of the morph file containing the perturbation to apply to the system.""")

lambda_val = Parameter("lambda_val", 0.0,
                       """Value of the lambda parameter at which to evaluate free energy gradients.""")

delta_lambda = Parameter("delta_lambda", 0.001,
                         """Value of the lambda interval used to evaluate free energy gradients by finite difference.""")

lambda_array = Parameter("lambda array",[] ,
                        """Array with all lambda values lambda_val needs to be part of the array. """)

shift_delta = Parameter("shift delta", 2.0,
                        """Value of the Lennard-Jones soft-core parameter.""")

coulomb_power = Parameter("coulomb power", 0,
                          """Value of the Coulombic soft-core parameter.""")

energy_frequency = Parameter("energy frequency", 1,
                             """The number of time steps between evaluation of free energy gradients.""")

simfile = Parameter("outdata_file", "simfile.dat", """Filename that records all output needed for the free energy analysis""")

perturbed_resnum = Parameter("perturbed residue number",1,"""The residue number of the molecule to morph.""")

verbose = Parameter("verbose", True, """Print debug output""")

from_QuBe = Parameter("from qube", True,
                                    """Whether or not to look for virtual sites at a QuBe-parameterised molecule.""")

def writeSystemData(system, moves, Trajectory, block, softcore_lambda=False):

    if softcore_lambda:
        if block == ncycles.val or block == 1:
            Trajectory.writeModel(system[MGName("all")], system.property("space"))
    else:
        if block % ncycles_per_snap.val == 0:
            if buffered_coords_freq.val > 0:
                dimensions = {}
                sysprops = system.propertyKeys()
                for prop in sysprops:
                    if prop.startswith("buffered_space"):
                        dimensions[str(prop)] = system.property(prop)
                Trajectory.writeBufferedModels(system[MGName("all")], dimensions)
            else:
                Trajectory.writeModel(system[MGName("all")], system.property("space"))

    moves_file = open("moves.dat", "w")
    print("%s" % moves, file=moves_file)
    moves_file.close()



def setupMoves(system, random_seed, GPUS):

    print("Setting up moves...")

    molecules = system[MGName("all")]

    Integrator_OpenMM = OpenMMMDIntegrator(molecules)

    Integrator_OpenMM.setPlatform(platform.val)
    Integrator_OpenMM.setConstraintType(constraint.val)
    Integrator_OpenMM.setCutoffType(cutoff_type.val)
    Integrator_OpenMM.setIntegrator(integrator_type.val)
    Integrator_OpenMM.setFriction(inverse_friction.val)  # Only meaningful for Langevin/Brownian integrators
    Integrator_OpenMM.setPrecision(precision.val)
    Integrator_OpenMM.setTimetoSkip(time_to_skip.val)

    Integrator_OpenMM.setDeviceIndex(str(GPUS))
    Integrator_OpenMM.setLJDispersion(lj_dispersion.val)

    if cutoff_type.val != "nocutoff":
        Integrator_OpenMM.setCutoffDistance(cutoff_dist.val)
    if cutoff_type.val == "cutoffperiodic":
        Integrator_OpenMM.setFieldDielectric(rf_dielectric.val)

    Integrator_OpenMM.setCMMremovalFrequency(cmm_removal.val)

    Integrator_OpenMM.setBufferFrequency(buffered_coords_freq.val)

    # if VirtualSite_flag.val:
    #     Integrator_OpenMM.setVirtualSite(True)

    if use_restraints.val:
        Integrator_OpenMM.setRestraint(True)

    if andersen.val:
        Integrator_OpenMM.setTemperature(temperature.val)
        Integrator_OpenMM.setAndersen(andersen.val)
        Integrator_OpenMM.setAndersenFrequency(andersen_frequency.val)

    if barostat.val:
        Integrator_OpenMM.setPressure(pressure.val)
        Integrator_OpenMM.setMCBarostat(barostat.val)
        Integrator_OpenMM.setMCBarostatFrequency(barostat_frequency.val)

    #print Integrator_OpenMM.getDeviceIndex()
    Integrator_OpenMM.initialise()

    mdmove = MolecularDynamics(molecules, Integrator_OpenMM, timestep.val,
                               {"velocity generator": MaxwellBoltzmann(temperature.val)})

    print("Created a MD move that uses OpenMM for all molecules on %s " % GPUS)

    moves = WeightedMoves()
    moves.add(mdmove, 1)

    if (not random_seed):
        random_seed = RanGenerator().randInt(100000, 1000000)
    print("Generated random seed number %d " % random_seed)

    moves.setGenerator(RanGenerator(random_seed))

    return moves


def setupDCD(system):
    r"""
    Parameters:
    ----------
    system : sire system
        sire system to be saved
    Return:
    ------
    trajectory : trajectory
   """
    files = os.listdir(os.getcwd())
    dcds = []
    for f in files:
        if f.endswith(".dcd"):
            dcds.append(f)

    dcds.sort()

    index = len(dcds) + 1

    dcd_filename = dcd_root.val + "%0009d" % index + ".dcd"
    softcore_almbda = True
    if lambda_val.val == 1.0 or lambda_val.val == 0.0:
        softcore_almbda = False
    if minimal_coordinate_saving.val and softcore_almbda:
        interval = ncycles.val*nmoves.val
        Trajectory = DCDFile(dcd_filename, system[MGName("all")], system.property("space"), timestep.val, interval)
    else:
        Trajectory = DCDFile(dcd_filename, system[MGName("all")], system.property("space"), timestep.val,
                         interval=buffered_coords_freq.val * ncycles_per_snap.val)

    return Trajectory



def createSystem(molecules):
    #print("Applying flexibility and zmatrix templates...")
    print("Creating the system...")

    moleculeNumbers = molecules.molNums()
    moleculeList = []

    for moleculeNumber in moleculeNumbers:
        molecule = molecules.molecule(moleculeNumber)[0].molecule()
        moleculeList.append(molecule)

    molecules = MoleculeGroup("molecules")
    ions = MoleculeGroup("ions")

    for molecule in moleculeList:
        natoms = molecule.nAtoms()
        if natoms == 1:
            ions.add(molecule)
        else:
            molecules.add(molecule)

    all = MoleculeGroup("all")
    all.add(molecules)
    all.add(ions)

    # Add these groups to the System
    system = System()

    system.add(all)
    system.add(molecules)
    system.add(ions)

    return system


def setupForcefields(system, space):

    print("Creating force fields... ")

    all = system[MGName("all")]
    molecules = system[MGName("molecules")]
    ions = system[MGName("ions")]

    # - first solvent-solvent coulomb/LJ (CLJ) energy
    internonbondedff = InterCLJFF("molecules:molecules")
    if (cutoff_type.val != "nocutoff"):
        internonbondedff.setUseReactionField(True)
        internonbondedff.setReactionFieldDielectric(rf_dielectric.val)
    internonbondedff.add(molecules)

    inter_ions_nonbondedff = InterCLJFF("ions:ions")
    if (cutoff_type.val != "nocutoff"):
        inter_ions_nonbondedff.setUseReactionField(True)
        inter_ions_nonbondedff.setReactionFieldDielectric(rf_dielectric.val)

    inter_ions_nonbondedff.add(ions)

    inter_ions_molecules_nonbondedff = InterGroupCLJFF("ions:molecules")
    if (cutoff_type.val != "nocutoff"):
        inter_ions_molecules_nonbondedff.setUseReactionField(True)
        inter_ions_molecules_nonbondedff.setReactionFieldDielectric(rf_dielectric.val)

    inter_ions_molecules_nonbondedff.add(ions, MGIdx(0))
    inter_ions_molecules_nonbondedff.add(molecules, MGIdx(1))

    # Now solute bond, angle, dihedral energy
    intrabondedff = InternalFF("molecules-intrabonded")
    intrabondedff.add(molecules)

    # Now solute intramolecular CLJ energy
    intranonbondedff = IntraCLJFF("molecules-intranonbonded")

    if (cutoff_type.val != "nocutoff"):
        intranonbondedff.setUseReactionField(True)
        intranonbondedff.setReactionFieldDielectric(rf_dielectric.val)

    intranonbondedff.add(molecules)

    # solute restraint energy
    #
    # We restrain atoms based ont he contents of the property "restrainedatoms"
    #
    restraintff = RestraintFF("restraint")

    if use_restraints.val:
        molnums = molecules.molecules().molNums()

        for molnum in molnums:
            mol = molecules.molecule(molnum)[0].molecule()
            try:
                mol_restrained_atoms = propertyToAtomNumVectorList(mol.property("restrainedatoms"))
            except UserWarning as error:
                error_type = re.search(r"(Sire\w*::\w*)", str(error)).group(0)
                if error_type == "SireBase::missing_property":
                    continue
                else:
                    raise error

            for restrained_line in mol_restrained_atoms:
                atnum = restrained_line[0]
                restraint_atom = mol.select(atnum)
                restraint_coords = restrained_line[1]
                restraint_k = restrained_line[2] * kcal_per_mol / (angstrom * angstrom)

                restraint = DistanceRestraint.harmonic(restraint_atom, restraint_coords, restraint_k)

                restraintff.add(restraint)

    # Here is the list of all forcefields
    forcefields = [internonbondedff, intrabondedff, intranonbondedff, inter_ions_nonbondedff,
                   inter_ions_molecules_nonbondedff, restraintff]

    for forcefield in forcefields:
        system.add(forcefield)

    system.setProperty("space", space)
    system.setProperty("switchingFunction", CHARMMSwitchingFunction(cutoff_dist.val))
    system.setProperty("combiningRules", VariantProperty(combining_rules.val))

    total_nrg = internonbondedff.components().total() + \
                intranonbondedff.components().total() + intrabondedff.components().total() + \
                inter_ions_nonbondedff.components().total() + inter_ions_molecules_nonbondedff.components().total() + \
                restraintff.components().total()

    e_total = system.totalComponent()

    system.setComponent(e_total, total_nrg)

    # Add a monitor that calculates the average total energy and average energy
    # deltas - we will collect both a mean average and an zwanzig average
    system.add("total_energy", MonitorComponent(e_total, Average()))

    return system


def setupForceFieldsFreeEnergy(system, space):
    r"""sets up the force field for the free energy calculation
    Parameters
    ----------
    system : Sire.system
    space : Sire.space
    Returns
    -------
    system : Sire.system
    """

    print ("Creating force fields... ")

    solutes = system[MGName("solutes")]

    solute = system[MGName("solute_ref")]
    solute_hard = system[MGName("solute_ref_hard")]
    solute_todummy = system[MGName("solute_ref_todummy")]
    solute_fromdummy = system[MGName("solute_ref_fromdummy")]

    solvent = system[MGName("solvent")]

    all = system[MGName("all")]

    # ''solvent'' is actually every molecule that isn't perturbed !
    solvent_intraff = InternalFF("solvent_intraff")
    solvent_intraff.add(solvent)

    # Solute bond, angle, dihedral energy
    solute_intraff = InternalFF("solute_intraff")
    solute_intraff.add(solute)

    # Solvent-solvent coulomb/LJ (CLJ) energy
    solventff = InterCLJFF("solvent:solvent")
    if (cutoff_type.val != "nocutoff"):
        solventff.setUseReactionField(True)
        solventff.setReactionFieldDielectric(rf_dielectric.val)
    solventff.add(solvent)

    #Solvent intramolecular CLJ energy
    solvent_intraclj = IntraCLJFF("solvent_intraclj")
    if (cutoff_type.val != "nocutoff"):
        solvent_intraclj.setUseReactionField(True)
        solvent_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solvent_intraclj.add(solvent)

    # Solute intramolecular CLJ energy
    solute_hard_intraclj = IntraCLJFF("solute_hard_intraclj")
    if (cutoff_type.val != "nocutoff"):
        solute_hard_intraclj.setUseReactionField(True)
        solute_hard_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_intraclj.add(solute_hard)

    solute_todummy_intraclj = IntraSoftCLJFF("solute_todummy_intraclj")
    solute_todummy_intraclj.setShiftDelta(shift_delta.val)
    solute_todummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_todummy_intraclj.setUseReactionField(True)
        solute_todummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_todummy_intraclj.add(solute_todummy)

    solute_fromdummy_intraclj = IntraSoftCLJFF("solute_fromdummy_intraclj")
    solute_fromdummy_intraclj.setShiftDelta(shift_delta.val)
    solute_fromdummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_fromdummy_intraclj.setUseReactionField(True)
        solute_fromdummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_fromdummy_intraclj.add(solute_fromdummy)

    solute_hard_todummy_intraclj = IntraGroupSoftCLJFF("solute_hard:todummy_intraclj")
    solute_hard_todummy_intraclj.setShiftDelta(shift_delta.val)
    solute_hard_todummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_hard_todummy_intraclj.setUseReactionField(True)
        solute_hard_todummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_todummy_intraclj.add(solute_hard, MGIdx(0))
    solute_hard_todummy_intraclj.add(solute_todummy, MGIdx(1))

    solute_hard_fromdummy_intraclj = IntraGroupSoftCLJFF("solute_hard:fromdummy_intraclj")
    solute_hard_fromdummy_intraclj.setShiftDelta(shift_delta.val)
    solute_hard_fromdummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_hard_fromdummy_intraclj.setUseReactionField(True)
        solute_hard_fromdummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_fromdummy_intraclj.add(solute_hard, MGIdx(0))
    solute_hard_fromdummy_intraclj.add(solute_fromdummy, MGIdx(1))

    solute_todummy_fromdummy_intraclj = IntraGroupSoftCLJFF("solute_todummy:fromdummy_intraclj")
    solute_todummy_fromdummy_intraclj.setShiftDelta(shift_delta.val)
    solute_todummy_fromdummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_todummy_fromdummy_intraclj.setUseReactionField(True)
        solute_todummy_fromdummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_todummy_fromdummy_intraclj.add(solute_todummy, MGIdx(0))
    solute_todummy_fromdummy_intraclj.add(solute_fromdummy, MGIdx(1))

    #Solute-solvent CLJ energy
    solute_hard_solventff = InterGroupCLJFF("solute_hard:solvent")
    if (cutoff_type.val != "nocutoff"):
        solute_hard_solventff.setUseReactionField(True)
        solute_hard_solventff.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_solventff.add(solute_hard, MGIdx(0))
    solute_hard_solventff.add(solvent, MGIdx(1))

    solute_todummy_solventff = InterGroupSoftCLJFF("solute_todummy:solvent")
    if (cutoff_type.val != "nocutoff"):
        solute_todummy_solventff.setUseReactionField(True)
        solute_todummy_solventff.setReactionFieldDielectric(rf_dielectric.val)
    solute_todummy_solventff.add(solute_todummy, MGIdx(0))
    solute_todummy_solventff.add(solvent, MGIdx(1))

    solute_fromdummy_solventff = InterGroupSoftCLJFF("solute_fromdummy:solvent")
    if (cutoff_type.val != "nocutoff"):
        solute_fromdummy_solventff.setUseReactionField(True)
        solute_fromdummy_solventff.setReactionFieldDielectric(rf_dielectric.val)
    solute_fromdummy_solventff.add(solute_fromdummy, MGIdx(0))
    solute_fromdummy_solventff.add(solvent, MGIdx(1))


    # TOTAL
    forcefields = [solute_intraff,
                   solute_hard_intraclj, solute_todummy_intraclj, solute_fromdummy_intraclj,
                   solute_hard_todummy_intraclj, solute_hard_fromdummy_intraclj,
                   solute_todummy_fromdummy_intraclj,
                   solvent_intraff,
                   solventff, solvent_intraclj,
                   solute_hard_solventff, solute_todummy_solventff, solute_fromdummy_solventff]


    for forcefield in forcefields:
        system.add(forcefield)

    system.setProperty("space", space)

    if (cutoff_type.val != "nocutoff"):
        system.setProperty("switchingFunction", CHARMMSwitchingFunction(cutoff_dist.val))
    else:
        system.setProperty("switchingFunction", NoCutoff())

    system.setProperty("combiningRules", VariantProperty(combining_rules.val))
    system.setProperty("coulombPower", VariantProperty(coulomb_power.val))
    system.setProperty("shiftDelta", VariantProperty(shift_delta.val))

    # TOTAL
    total_nrg = solute_intraff.components().total() + solute_hard_intraclj.components().total() + \
                solute_todummy_intraclj.components().total(0) + solute_fromdummy_intraclj.components().total(0) + \
                solute_hard_todummy_intraclj.components().total(
                    0) + solute_hard_fromdummy_intraclj.components().total(0) + \
                solute_todummy_fromdummy_intraclj.components().total(0) + \
                solvent_intraff.components().total() + solventff.components().total() + \
                solvent_intraclj.components().total() + \
                solute_hard_solventff.components().total() + \
                solute_todummy_solventff.components().total(0) + \
                solute_fromdummy_solventff.components().total(0)

    e_total = system.totalComponent()

    lam = Symbol("lambda")

    system.setComponent(e_total, total_nrg)

    system.setConstant(lam, 0.0)

    system.add(PerturbationConstraint(solutes))

    # NON BONDED Alpha constraints for the soft force fields

    system.add(PropertyConstraint("alpha0", FFName("solute_todummy_intraclj"), lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_fromdummy_intraclj"), 1 - lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_hard:todummy_intraclj"), lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_hard:fromdummy_intraclj"), 1 - lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_todummy:fromdummy_intraclj"), Max(lam, 1 - lam)))
    system.add(PropertyConstraint("alpha0", FFName("solute_todummy:solvent"), lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_fromdummy:solvent"), 1 - lam))

    system.setComponent(lam, lambda_val.val)

    # printEnergies( system.componentValues() )

    return system


def vsiteListToProperty(list):
    prop = Properties()
    i = 0
    for entry in list:
        for key, value in entry.items():
            prop.setProperty("%s(%d)" % (key,i), VariantProperty(value))
        i += 1
    prop.setProperty("nvirtualsites",VariantProperty(i))
    return prop

# def ffToProperty(string):
#     prop = Properties()      
#     prop.setProperty("forcefield",VariantProperty("qube"))
#     return prop



def setupForcefields(system, space):

    print("Creating force fields... ")

    all = system[MGName("all")]
    #print("all = ", all)
    molecules = system[MGName("molecules")]
    #print("molecules = ", molecules,)
    ions = system[MGName("ions")]
    #print("ions = ", ions)
    
    # - first solvent-solvent coulomb/LJ (CLJ) energy
    internonbondedff = InterCLJFF("molecules:molecules")
    if (cutoff_type.val != "nocutoff"):
        internonbondedff.setUseReactionField(True)
        internonbondedff.setReactionFieldDielectric(rf_dielectric.val)
    internonbondedff.add(molecules)
    #print("internonbondedff = ", internonbondedff)

    inter_ions_nonbondedff = InterCLJFF("ions:ions")
    if (cutoff_type.val != "nocutoff"):
        inter_ions_nonbondedff.setUseReactionField(True)
        inter_ions_nonbondedff.setReactionFieldDielectric(rf_dielectric.val)

    inter_ions_nonbondedff.add(ions)

    inter_ions_molecules_nonbondedff = InterGroupCLJFF("ions:molecules")
    if (cutoff_type.val != "nocutoff"):
        inter_ions_molecules_nonbondedff.setUseReactionField(True)
        inter_ions_molecules_nonbondedff.setReactionFieldDielectric(rf_dielectric.val)

    inter_ions_molecules_nonbondedff.add(ions, MGIdx(0))
    inter_ions_molecules_nonbondedff.add(molecules, MGIdx(1))

    # Now solute bond, angle, dihedral energy
    intrabondedff = InternalFF("molecules-intrabonded")
    intrabondedff.add(molecules)

    # Now solute intramolecular CLJ energy
    intranonbondedff = IntraCLJFF("molecules-intranonbonded")

    if (cutoff_type.val != "nocutoff"):
        intranonbondedff.setUseReactionField(True)
        intranonbondedff.setReactionFieldDielectric(rf_dielectric.val)

    intranonbondedff.add(molecules)

    # solute restraint energy
    #
    # We restrain atoms based ont he contents of the property "restrainedatoms"
    #
    restraintff = RestraintFF("restraint")

    if use_restraints.val:
        molnums = molecules.molecules().molNums()

        for molnum in molnums:
            mol = molecules.molecule(molnum)[0].molecule()
            try:
                mol_restrained_atoms = propertyToAtomNumVectorList(mol.property("restrainedatoms"))
            except UserWarning as error:
                error_type = re.search(r"(Sire\w*::\w*)", str(error)).group(0)
                if error_type == "SireBase::missing_property":
                    continue
                else:
                    raise error

            for restrained_line in mol_restrained_atoms:
                atnum = restrained_line[0]
                restraint_atom = mol.select(atnum)
                restraint_coords = restrained_line[1]
                restraint_k = restrained_line[2] * kcal_per_mol / (angstrom * angstrom)

                restraint = DistanceRestraint.harmonic(restraint_atom, restraint_coords, restraint_k)

                restraintff.add(restraint)

    # Here is the list of all forcefields
    forcefields = [internonbondedff, intrabondedff, intranonbondedff, inter_ions_nonbondedff,
                   inter_ions_molecules_nonbondedff, restraintff]

    for forcefield in forcefields:
        system.add(forcefield)

    system.setProperty("space", space)
    system.setProperty("switchingFunction", CHARMMSwitchingFunction(cutoff_dist.val))
    system.setProperty("combiningRules", VariantProperty(combining_rules.val))

    total_nrg = internonbondedff.components().total() + \
                intranonbondedff.components().total() + intrabondedff.components().total() + \
                inter_ions_nonbondedff.components().total() + inter_ions_molecules_nonbondedff.components().total() + \
                restraintff.components().total()

    e_total = system.totalComponent()

    system.setComponent(e_total, total_nrg)

    # Add a monitor that calculates the average total energy and average energy
    # deltas - we will collect both a mean average and an zwanzig average
    system.add("total_energy", MonitorComponent(e_total, Average()))

    return system


def createSystemFreeEnergy(molecules):
    r"""creates the system for free energy calculation
    Parameters
    ----------
    molecules : Sire.molecules
        Sire object that contains a lot of information about molecules
    Returns
    -------
    system : Sire.system
    """
    print ("Create the System...")

    moleculeNumbers = molecules.molNums()
    moleculeList = []

    for moleculeNumber in moleculeNumbers:
        molecule = molecules.molecule(moleculeNumber)[0].molecule()
        moleculeList.append(molecule)

    # Scan input to find a molecule with passed residue number 
    # The residue name of the first residue in this molecule is
    # used to name the solute. This is used later to match
    # templates in the flex/pert files.

    solute = None
    for molecule in moleculeList:
        if ( molecule.residue(ResIdx(0)).number() == ResNum(perturbed_resnum.val) ):
            solute = molecule
            moleculeList.remove(molecule)
            break

    if solute is None:
        print ("FATAL ! Could not find a solute to perturb with residue number %s in the input ! Check the value of your cfg keyword 'perturbed residue number'" % perturbed_resnum.val)
        sys.exit(-1)

    #solute = moleculeList[0]

    lig_name = solute.residue(ResIdx(0)).name().value()

    solute = solute.edit().rename(lig_name).commit()

    perturbations_lib = PerturbationsLibrary(morphfile.val)
    solute = perturbations_lib.applyTemplate(solute)

    perturbations = solute.property("perturbations")

    lam = Symbol("lambda")

    initial = Perturbation.symbols().initial()
    final = Perturbation.symbols().final()

    solute = solute.edit().setProperty("perturbations",
                                       perturbations.recreate((1 - lam) * initial + lam * final)).commit()

    # We put atoms in three groups depending on what happens in the perturbation
    # non dummy to non dummy --> the hard group, uses a normal intermolecular FF
    # non dummy to dummy --> the todummy group, uses SoftFF with alpha = Lambda
    # dummy to non dummy --> the fromdummy group, uses SoftFF with alpha = 1 - Lambda
    # We start assuming all atoms are hard atoms. Then we call getDummies to find which atoms
    # start/end as dummies and update the hard, todummy and fromdummy groups accordingly

    solute_grp_ref = MoleculeGroup("solute_ref", solute)
    solute_grp_ref_hard = MoleculeGroup("solute_ref_hard")
    solute_grp_ref_todummy = MoleculeGroup("solute_ref_todummy")
    solute_grp_ref_fromdummy = MoleculeGroup("solute_ref_fromdummy")

    solute_ref_hard = solute.selectAllAtoms()
    solute_ref_todummy = solute_ref_hard.invert()
    solute_ref_fromdummy = solute_ref_hard.invert()

    to_dummies, from_dummies = getDummies(solute)

    if to_dummies is not None:
        ndummies = to_dummies.count()
        dummies = to_dummies.atoms()

        for x in range(0, ndummies):
            dummy_index = dummies[x].index()
            solute_ref_hard = solute_ref_hard.subtract(solute.select(dummy_index))
            solute_ref_todummy = solute_ref_todummy.add(solute.select(dummy_index))

    if from_dummies is not None:
        ndummies = from_dummies.count()
        dummies = from_dummies.atoms()

        for x in range(0, ndummies):
            dummy_index = dummies[x].index()
            solute_ref_hard = solute_ref_hard.subtract(solute.select(dummy_index))
            solute_ref_fromdummy = solute_ref_fromdummy.add(solute.select(dummy_index))

    solute_grp_ref_hard.add(solute_ref_hard)
    solute_grp_ref_todummy.add(solute_ref_todummy)
    solute_grp_ref_fromdummy.add(solute_ref_fromdummy)

    solutes = MoleculeGroup("solutes")
    solutes.add(solute)

    molecules = MoleculeGroup("molecules")
    molecules.add(solute)

    solvent = MoleculeGroup("solvent")

    #for molecule in moleculeList[1:]:
    for molecule in moleculeList:
        molecules.add(molecule)
        solvent.add(molecule)

    all = MoleculeGroup("all")

    all.add(molecules)
    all.add(solvent)

    all.add(solutes)
    all.add(solute_grp_ref)
    all.add(solute_grp_ref_hard)
    all.add(solute_grp_ref_todummy)
    all.add(solute_grp_ref_fromdummy)

    # Add these groups to the System
    system = System()

    system.add(solutes)
    system.add(solute_grp_ref)
    system.add(solute_grp_ref_hard)
    system.add(solute_grp_ref_todummy)
    system.add(solute_grp_ref_fromdummy)

    system.add(molecules)

    system.add(solvent)

    system.add(all)

    return system


def setupForceFieldsFreeEnergy(system, space):
    r"""sets up the force field for the free energy calculation
    Parameters
    ----------
    system : Sire.system
    space : Sire.space
    Returns
    -------
    system : Sire.system
    """

    print ("Creating force fields... ")

    solutes = system[MGName("solutes")]

    solute = system[MGName("solute_ref")]
    solute_hard = system[MGName("solute_ref_hard")]
    solute_todummy = system[MGName("solute_ref_todummy")]
    solute_fromdummy = system[MGName("solute_ref_fromdummy")]

    solvent = system[MGName("solvent")]

    all = system[MGName("all")]

    # ''solvent'' is actually every molecule that isn't perturbed !
    solvent_intraff = InternalFF("solvent_intraff")
    solvent_intraff.add(solvent)

    # Solute bond, angle, dihedral energy
    solute_intraff = InternalFF("solute_intraff")
    solute_intraff.add(solute)

    # Solvent-solvent coulomb/LJ (CLJ) energy
    solventff = InterCLJFF("solvent:solvent")
    if (cutoff_type.val != "nocutoff"):
        solventff.setUseReactionField(True)
        solventff.setReactionFieldDielectric(rf_dielectric.val)
    solventff.add(solvent)

    #Solvent intramolecular CLJ energy
    solvent_intraclj = IntraCLJFF("solvent_intraclj")
    if (cutoff_type.val != "nocutoff"):
        solvent_intraclj.setUseReactionField(True)
        solvent_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solvent_intraclj.add(solvent)

    # Solute intramolecular CLJ energy
    solute_hard_intraclj = IntraCLJFF("solute_hard_intraclj")
    if (cutoff_type.val != "nocutoff"):
        solute_hard_intraclj.setUseReactionField(True)
        solute_hard_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_intraclj.add(solute_hard)

    solute_todummy_intraclj = IntraSoftCLJFF("solute_todummy_intraclj")
    solute_todummy_intraclj.setShiftDelta(shift_delta.val)
    solute_todummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_todummy_intraclj.setUseReactionField(True)
        solute_todummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_todummy_intraclj.add(solute_todummy)

    solute_fromdummy_intraclj = IntraSoftCLJFF("solute_fromdummy_intraclj")
    solute_fromdummy_intraclj.setShiftDelta(shift_delta.val)
    solute_fromdummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_fromdummy_intraclj.setUseReactionField(True)
        solute_fromdummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_fromdummy_intraclj.add(solute_fromdummy)

    solute_hard_todummy_intraclj = IntraGroupSoftCLJFF("solute_hard:todummy_intraclj")
    solute_hard_todummy_intraclj.setShiftDelta(shift_delta.val)
    solute_hard_todummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_hard_todummy_intraclj.setUseReactionField(True)
        solute_hard_todummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_todummy_intraclj.add(solute_hard, MGIdx(0))
    solute_hard_todummy_intraclj.add(solute_todummy, MGIdx(1))

    solute_hard_fromdummy_intraclj = IntraGroupSoftCLJFF("solute_hard:fromdummy_intraclj")
    solute_hard_fromdummy_intraclj.setShiftDelta(shift_delta.val)
    solute_hard_fromdummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_hard_fromdummy_intraclj.setUseReactionField(True)
        solute_hard_fromdummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_fromdummy_intraclj.add(solute_hard, MGIdx(0))
    solute_hard_fromdummy_intraclj.add(solute_fromdummy, MGIdx(1))

    solute_todummy_fromdummy_intraclj = IntraGroupSoftCLJFF("solute_todummy:fromdummy_intraclj")
    solute_todummy_fromdummy_intraclj.setShiftDelta(shift_delta.val)
    solute_todummy_fromdummy_intraclj.setCoulombPower(coulomb_power.val)
    if (cutoff_type.val != "nocutoff"):
        solute_todummy_fromdummy_intraclj.setUseReactionField(True)
        solute_todummy_fromdummy_intraclj.setReactionFieldDielectric(rf_dielectric.val)
    solute_todummy_fromdummy_intraclj.add(solute_todummy, MGIdx(0))
    solute_todummy_fromdummy_intraclj.add(solute_fromdummy, MGIdx(1))

    #Solute-solvent CLJ energy
    solute_hard_solventff = InterGroupCLJFF("solute_hard:solvent")
    if (cutoff_type.val != "nocutoff"):
        solute_hard_solventff.setUseReactionField(True)
        solute_hard_solventff.setReactionFieldDielectric(rf_dielectric.val)
    solute_hard_solventff.add(solute_hard, MGIdx(0))
    solute_hard_solventff.add(solvent, MGIdx(1))

    solute_todummy_solventff = InterGroupSoftCLJFF("solute_todummy:solvent")
    if (cutoff_type.val != "nocutoff"):
        solute_todummy_solventff.setUseReactionField(True)
        solute_todummy_solventff.setReactionFieldDielectric(rf_dielectric.val)
    solute_todummy_solventff.add(solute_todummy, MGIdx(0))
    solute_todummy_solventff.add(solvent, MGIdx(1))

    solute_fromdummy_solventff = InterGroupSoftCLJFF("solute_fromdummy:solvent")
    if (cutoff_type.val != "nocutoff"):
        solute_fromdummy_solventff.setUseReactionField(True)
        solute_fromdummy_solventff.setReactionFieldDielectric(rf_dielectric.val)
    solute_fromdummy_solventff.add(solute_fromdummy, MGIdx(0))
    solute_fromdummy_solventff.add(solvent, MGIdx(1))


    # TOTAL
    forcefields = [solute_intraff,
                   solute_hard_intraclj, solute_todummy_intraclj, solute_fromdummy_intraclj,
                   solute_hard_todummy_intraclj, solute_hard_fromdummy_intraclj,
                   solute_todummy_fromdummy_intraclj,
                   solvent_intraff,
                   solventff, solvent_intraclj,
                   solute_hard_solventff, solute_todummy_solventff, solute_fromdummy_solventff]


    for forcefield in forcefields:
        system.add(forcefield)

    system.setProperty("space", space)

    if (cutoff_type.val != "nocutoff"):
        system.setProperty("switchingFunction", CHARMMSwitchingFunction(cutoff_dist.val))
    else:
        system.setProperty("switchingFunction", NoCutoff())

    system.setProperty("combiningRules", VariantProperty(combining_rules.val))
    system.setProperty("coulombPower", VariantProperty(coulomb_power.val))
    system.setProperty("shiftDelta", VariantProperty(shift_delta.val))

    # TOTAL
    total_nrg = solute_intraff.components().total() + solute_hard_intraclj.components().total() + \
                solute_todummy_intraclj.components().total(0) + solute_fromdummy_intraclj.components().total(0) + \
                solute_hard_todummy_intraclj.components().total(
                    0) + solute_hard_fromdummy_intraclj.components().total(0) + \
                solute_todummy_fromdummy_intraclj.components().total(0) + \
                solvent_intraff.components().total() + solventff.components().total() + \
                solvent_intraclj.components().total() + \
                solute_hard_solventff.components().total() + \
                solute_todummy_solventff.components().total(0) + \
                solute_fromdummy_solventff.components().total(0)

    e_total = system.totalComponent()

    lam = Symbol("lambda")

    system.setComponent(e_total, total_nrg)

    system.setConstant(lam, 0.0)

    system.add(PerturbationConstraint(solutes))

    # NON BONDED Alpha constraints for the soft force fields

    system.add(PropertyConstraint("alpha0", FFName("solute_todummy_intraclj"), lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_fromdummy_intraclj"), 1 - lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_hard:todummy_intraclj"), lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_hard:fromdummy_intraclj"), 1 - lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_todummy:fromdummy_intraclj"), Max(lam, 1 - lam)))
    system.add(PropertyConstraint("alpha0", FFName("solute_todummy:solvent"), lam))
    system.add(PropertyConstraint("alpha0", FFName("solute_fromdummy:solvent"), 1 - lam))

    system.setComponent(lam, lambda_val.val)

    # printEnergies( system.componentValues() )

    return system


def setupMovesFreeEnergy(system, random_seed, GPUS, lambda_val):

    print ("Setting up moves...")

    molecules = system[MGName("molecules")]
    solute = system[MGName("solute_ref")]
    solute_hard = system[MGName("solute_ref_hard")]
    solute_todummy = system[MGName("solute_ref_todummy")]
    solute_fromdummy = system[MGName("solute_ref_fromdummy")]

    Integrator_OpenMM = OpenMMFrEnergyST(molecules, solute, solute_hard, solute_todummy, solute_fromdummy)
    Integrator_OpenMM.setRandomSeed(random_seed)
    Integrator_OpenMM.setIntegrator(integrator_type.val)
    Integrator_OpenMM.setFriction(inverse_friction.val)  # Only meaningful for Langevin/Brownian integrators
    Integrator_OpenMM.setPlatform(platform.val)
    Integrator_OpenMM.setConstraintType(constraint.val)
    Integrator_OpenMM.setCutoffType(cutoff_type.val)
    Integrator_OpenMM.setFieldDielectric(rf_dielectric.val)
    Integrator_OpenMM.setAlchemicalValue(lambda_val.val)
    Integrator_OpenMM.setAlchemicalArray(lambda_array.val)
    Integrator_OpenMM.setDeviceIndex(str(GPUS))
    Integrator_OpenMM.setCoulombPower(coulomb_power.val)
    Integrator_OpenMM.setShiftDelta(shift_delta.val)
    Integrator_OpenMM.setDeltatAlchemical(delta_lambda.val)
    Integrator_OpenMM.setPrecision(precision.val)
    Integrator_OpenMM.setTimetoSkip(time_to_skip.val)
    Integrator_OpenMM.setBufferFrequency(buffered_coords_freq.val)

    if cutoff_type != "nocutoff":
        Integrator_OpenMM.setCutoffDistance(cutoff_dist.val)

    Integrator_OpenMM.setCMMremovalFrequency(cmm_removal.val)

    Integrator_OpenMM.setEnergyFrequency(energy_frequency.val)

    if use_restraints.val:
        Integrator_OpenMM.setRestraint(True)

    if andersen.val:
        Integrator_OpenMM.setTemperature(temperature.val)
        Integrator_OpenMM.setAndersen(andersen.val)
        Integrator_OpenMM.setAndersenFrequency(andersen_frequency.val)

    if barostat.val:
        Integrator_OpenMM.setPressure(pressure.val)
        Integrator_OpenMM.setMCBarostat(barostat.val)
        Integrator_OpenMM.setMCBarostatFrequency(barostat_frequency.val)

    #This calls the OpenMMFrEnergyST initialise function
    Integrator_OpenMM.initialise()
    velocity_generator = MaxwellBoltzmann(temperature.val)
    velocity_generator.setGenerator(RanGenerator(random_seed))

    mdmove = MolecularDynamics(molecules, Integrator_OpenMM, timestep.val,
                              {"velocity generator":velocity_generator})

    print("Created a MD move that uses OpenMM for all molecules on %s " % GPUS)

    moves = WeightedMoves()
    moves.add(mdmove, 1)

    if (not random_seed):
        random_seed = RanGenerator().randInt(100000, 1000000)

    print("Generated random seed number %d " % random_seed)

    moves.setGenerator(RanGenerator(random_seed))

    return moves



def readXmlParameters(pdbfile, xmlfile):
# 1) Read a pdb file describing the system to simulate

    p = PDB2(pdbfile)
    s = p.toSystem()
    molecules = s.molecules()
    #print (molecules)
    with open (pdbfile, "r") as f:
        for line in f:
            if line.split()[0] == "CRYST1" :
                print (line)
                pbc_x = float(line.split()[1])
                pbc_y = float(line.split()[2])
                pbc_z = float(line.split()[3])
                space = PeriodicBox(Vector(pbc_x, pbc_y, pbc_z))
                break
            else:
                space = Cartesian()
    #print("space:", space)

    system = System()

     # 2) Now we read the xml file, and store parameters for each molecule


    import xml.dom.minidom as minidom
    xmldoc = minidom.parse(xmlfile)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: TYPE ~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_type = xmldoc.getElementsByTagName('Type')
    dicts_type = []
    for items in itemlist_type:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_type.append(d)
    dicts_tp =  str(dicts_type).split()
    #print (dicts_tp)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: ATOM ~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_atom = xmldoc.getElementsByTagName('Atom')
    dicts_atom = []
    for items in itemlist_atom:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_atom.append(d)
    dicts_at =  str(dicts_atom).split()
    #print (dicts_at)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: BOND ~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_bond = xmldoc.getElementsByTagName('Bond')
    dicts_bond = []
    for items in itemlist_bond:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_bond.append(d)
    dicts_b =  str(dicts_bond).split()
    #print (dicts_b)

    nbond = itemlist_bond.length

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: ANGLE ~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_angle = xmldoc.getElementsByTagName('Angle')
    dicts_angle = []
    for items in itemlist_angle:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_angle.append(d)
    dicts_ang =  str(dicts_angle).split()
    print (dicts_angle)

    nAngles= itemlist_angle.length
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: PROPER ~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_proper = xmldoc.getElementsByTagName('Proper')
    dicts_proper = []
    for items in itemlist_proper:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_proper.append(d)
    dicts_pr =  str(dicts_proper).split()
    #print (dicts_pr)

    nProper = itemlist_proper.length

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: IMPROPER ~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_improper = xmldoc.getElementsByTagName('Improper')
    dicts_improper = []
    for items in itemlist_improper:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_improper.append(d)
    dicts_impr =  str(dicts_improper).split()
    #print (dicts_impr)
    nImproper = itemlist_improper.length

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: VIRTUAL SITES ~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_VirtualSite = xmldoc.getElementsByTagName('VirtualSite')
    dicts_virtualsite = []
    for items in itemlist_VirtualSite:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_virtualsite.append(d)
    #dicts_vs =  str(dicts_virtualsite).split()
    #print (dicts_vs)
    nVirtualSites = itemlist_VirtualSite.length 


    v_site_CLJ = []
    for i in range(0, int(len(dicts_atom))):
        if dicts_atom[i]['type'][0] == 'v':
            v_site_CLJ = dicts_atom[i]
            dicts_virtualsite.append(v_site_CLJ)

    for i in range(0, len(itemlist_VirtualSite)):
        dicts_virtualsite[i].update(dicts_virtualsite[i+len(itemlist_VirtualSite)])
        dicts_virtualsite[i].update(dicts_virtualsite[i+2*len(itemlist_VirtualSite)]) 

    dict_vs = []
    for i in range(0, len(itemlist_VirtualSite)):
        dicts_virtualsite[i]
        dict_vs.append(dicts_virtualsite[i])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~ TAG NAME: RESIDUE ~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_residue = xmldoc.getElementsByTagName('Residue')
    dicts_residue = []
    for items in itemlist_residue:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_residue.append(d)
    dicts_res =  str(dicts_residue).split()
    #print (dicts_res)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~ TAG NAME: NON BONDED FORCE ~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    itemlist_nonbond = xmldoc.getElementsByTagName('NonbondedForce')
    dicts_nonb = []
    for items in itemlist_nonbond:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_nonb.append(d)
    dicts_nb =  str(dicts_nonb).split()
    #print (dicts_nb)
    nNonBonded = itemlist_nonbond.length

    # 3) Now we create an Amberparameters object for each molecule
    molnums = molecules.molNums()

    newmolecules = Molecules()
    for molnum in molnums:
        mol = molecules.at(molnum)
        #print (mol) 
        


        # Add potential virtual site parameters
        if len(dicts_virtualsite) > 0:
            mol = mol.edit().setProperty("virtual-sites", vsiteListToProperty(dict_vs)).commit()

            
        # We populate the Amberparameters object with a list of bond, angle, dihedrals
        # We look up parameters from the contents of the xml file
        # We also have to set the atomic parameters (q, sigma, epsilon)

        editmol = mol.edit()
        mol_params = AmberParameters(editmol) #SireMol::AmberParameters()
        atoms = editmol.atoms()
        # We update atom parameters see setAtomParameters in SireIO/amber.cpp l2122 
        natoms = editmol.nAtoms()
        #print("number of atoms is %s" %natoms)
        
    #natoms don't include the virtual sites! 

    # Loop over each molecule in the molecules object


        opls=[]
        for i in range (0, int(len(dicts_atom)/2)): 
            opl={} 
            opl = dicts_atom[i]['type'] 
            opls.append(opl) 

        name=[]
        for i in range (0, int(len(dicts_atom)/2)): 
            nm={} 
            nm = dicts_atom[i]['name'] 
            name.append(nm) 


        two=[] 
        #print(len(name)) 
        for i in range(0, len(name)): 
            t=(opls[i],name[i]) 
            two.append(t) 

        import numpy as np 

        atom_sorted = []
        for j in range(0, len(two)): 
            for i in range(int(len(dicts_atom)/2), len(dicts_atom)):   
                if dicts_atom[i]['type'] == two[j][0]: 
                    dic_a = {}
                    dic_a = dicts_atom[i]
                    atom_sorted.append(dic_a)
      
        type_sorted = []
        for j in range(0, len(two)): 
            for i in range(0, int(len(dicts_type))):   
                if dicts_type[i]['name'] == two[j][0]: 
                    dic_t = {}
                    dic_t = dicts_type[i]
                    type_sorted.append(dic_t)
        print(" ")
        print("There are ",natoms," atoms in this molecule. ")
        print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")

        for atom in atoms: 
            editatom = editmol.atom(atom.index())

            i = int(str(atom.number()).split('(')[1].replace(")" , " ")) 

            editatom.setProperty("charge", float(atom_sorted[i-1]['charge']) * mod_electron)
            editatom.setProperty("mass", float(type_sorted[i-1]['mass']) * g_per_mol) 
            editatom.setProperty("LJ", LJParameter( float(atom_sorted[i-1]['sigma'])*10 * angstrom , float(atom_sorted[i-1]['epsilon'])/4.184 * kcal_per_mol))
            editatom.setProperty("ambertype", dicts_atom[i-1]['type'])
           
            editmol = editatom.molecule()

        # Now we create a connectivity see setConnectivity in SireIO/amber.cpp l2144
        # XML data tells us how atoms are bonded in the molecule (Bond 'from' and 'to')
        
        if natoms > 1:
            print("Set up connectivity")

            con = []
            for i in range(0,int(nbond/2)):
                if natoms > 1:   
                    connect_prop= {}
                    connect_prop = dicts_bond[i]['from'], dicts_bond[i]['to']
                con.append(connect_prop)
            

            conn = Connectivity(editmol.info()).edit()

            for j in range(0,len(con)):
                conn.connect(atoms[int(con[j][0]) ].index(), atoms[int(con[j][1]) ].index()) 
                   
            
            editmol.setProperty("connectivity", conn.commit()).commit()
            mol = editmol.setProperty("connectivity", conn.commit()).commit()
            system.update(mol)

             # Now we add bond parameters to the Sire molecule. We also update amberparameters see SireIO/amber.cpp l2154

            internalff = InternalFF()
                    
            bondfuncs = TwoAtomFunctions(mol)
            r = internalff.symbols().bond().r()

            for j in range(0,len(con)):
                bondfuncs.set(atoms[int(con[j][0]) ].index(), atoms[int(con[j][1]) ].index(), float(dicts_bond[j+len(con)]['k'])/(2*100*4.184)* (float(dicts_bond[j+len(con)]['length'])*10 - r) **2  )
                bond_id = BondID(atoms[int(con[j][0])].index(), atoms[int(con[j][1])].index())
                mol_params.add(bond_id, float(dicts_bond[j+len(con)]['k'])/(2*100*4.184), float(dicts_bond[j+len(con)]['length'])*10 )   
                editmol.setProperty("bonds", bondfuncs).commit()
                molecule = editmol.commit()
             

            mol_params.getAllBonds() 

            editmol.setProperty("amberparameters", mol_params).commit() # Weird, should work - investigate ? 
            molecule = editmol.commit()

        # Now we add angle parameters to the Sire molecule. We also update amberparameters see SireIO/amber.cpp L2172
        if natoms > 2:
            print("Set up angles")

            anglefuncs = ThreeAtomFunctions(mol)
            at1 = []
            for i in range(0, nAngles):
                a1 = {}
                to_str1 = str(re.findall(r"\d+",str(dicts_angle[i]['class1'])))
                if dicts_atom[i]['type'][0] == 'o': #if opls_
                    a1 = int(to_str1.replace("[","").replace("]","").replace("'","") )-800

                    at1.append(a1)
          
            print(at1)
            at2 = []
            for i in range(0, nAngles):
                a2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_angle[i]['class2'])))
                if dicts_atom[i]['type'][0] == 'o': #if opls_
                    a2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                    at2.append(a2)
         

            at3 = []
            for i in range(0, nAngles):
                a3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_angle[i]['class3'])))
                if dicts_atom[i]['type'][0] == 'o': #if opls_
                    a3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800

                    at3.append(a3)
            print ("lengths:", len(at1), len(at2), len(at3))
            print("number of Angles =",nAngles)
            theta = internalff.symbols().angle().theta()
            for j in range(0,nAngles- nVirtualSites):

                anglefuncs.set( atoms[at1[j]].index(), atoms[at2[j]].index(), atoms[at3[j]].index(), float(dicts_angle[j]['k'])/(2*4.184) * ( (float(dicts_angle[j]['angle']) - theta )**2 ))
                angle_id = AngleID( atoms[int(at1[j])].index(), atoms[int(at2[j])].index(), atoms[int(at3[j])].index())
                mol_params.add(angle_id, float(dicts_angle[j]['k'])/(2*4.184), float(dicts_angle[j]['angle']) ) 
            
        # Now we add dihedral parameters to the Sire molecule. We also update amberparameters see SireIO/amber.cpp L2190

        if natoms > 3:
            print("Set up dihedrals")
            di1 = []

            for i in range(0, nProper):
                d1 = {}
                to_str1 = str(re.findall(r"\d+",str(dicts_proper[i]['class1'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d1 = int(to_str1.replace("[","").replace("]","").replace("'","") )-800
                elif dicts_atom[i]['type'][0] == 'Q':  #if QUBE_
                    d1 = int(to_str1.replace("[","").replace("]","").replace("'","") )
               
                di1.append(d1)


            di2 = []
            for i in range(0, nProper):
                d2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_proper[i]['class2'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800

                    di2.append(d2)

            di3 = []
            for i in range(0, nProper):
                d3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_proper[i]['class3'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800

                    di3.append(d3)
          

            di4 = []
            for i in range(0, nProper):
                d4 = {}
                to_str4 = str(re.findall(r"\d+",str(dicts_proper[i]['class4'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )-800

                    di4.append(d4)
 

            dihedralfuncs = FourAtomFunctions(mol)
    
            phi = internalff.symbols().dihedral().phi()
            for i in range(0,nProper):  
                if atoms[int(di1[i])].index() != atoms[int(di4[i])].index():
                    dihedral_id = DihedralID( atoms[int(di1[i])].index(), atoms[int(di2[i])].index(), atoms[int(di3[i])].index(), atoms[int(di4[i])].index()) 
                    dih1= float(dicts_proper[i]['k1'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity1'])* phi- float(dicts_proper[i]['phase1'])))
                    dih2= float(dicts_proper[i]['k2'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity2'])* phi- float(dicts_proper[i]['phase2'])))
                    dih3= float(dicts_proper[i]['k3'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity3'])* phi- float(dicts_proper[i]['phase3'])))
                    dih4= float(dicts_proper[i]['k4'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity4'])* phi- float(dicts_proper[i]['phase4'])))
                    dih_fun = dih1 + dih2 +dih3 +dih4
                    dihedralfuncs.set(dihedral_id, dih_fun)

                    for t in range(1,5):
                        mol_params.add(dihedral_id, float(dicts_proper[i]['k%s'%t])/4.184, int(dicts_proper[i]['periodicity%s'%t]), float(dicts_proper[i]['phase%s'%t]) ) 
            

            print("Set up impropers")

            di_im1 = []
            for i in range(0, nImproper):
                d1 = {}
                to_str1 = str(re.findall(r"\d+",str(dicts_improper[i]['class1'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d1 = int(to_str1.replace("[","").replace("]","").replace("'","") )-800

                    di_im1.append(d1)


            di_im2 = []
            for i in range(0, nImproper):
                d2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_improper[i]['class2'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                elif dicts_atom[i]['type'][0] == 'Q':
                    d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )

                
                di_im2.append(d2)

            di_im3 = []
            for i in range(0, nImproper):
                d3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_improper[i]['class3'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800
                elif dicts_atom[i]['type'][0] == 'Q':
                    d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )
                
                di_im3.append(d3)

            di_im4 = []
            for i in range(0, nImproper):
                d4 = {}
                to_str4 = str(re.findall(r"\d+",str(dicts_improper[i]['class4'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )-800
                elif dicts_atom[i]['type'][0] == 'Q':
                    d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )
                
                di_im4.append(d4)
            
            improperfuncs = FourAtomFunctions(mol)

            phi_im = internalff.symbols().improper().phi()


            for i in range(0,nImproper):  
                improper_id = ImproperID( atoms[int(di_im2[i])].index(), atoms[int(di_im3[i])].index(), atoms[int(di_im1[i])].index(), atoms[int(di_im4[i])].index()) 
                imp1= float(dicts_improper[i]['k1'])*(1/4.184)*(1+Cos(int(dicts_improper[i]['periodicity1'])* phi_im - float(dicts_improper[i]['phase1'])))
                imp2= float(dicts_improper[i]['k2'])*(1/4.184)*(1+Cos(int(dicts_improper[i]['periodicity2'])* phi_im - float(dicts_improper[i]['phase2'])))
                imp3= float(dicts_improper[i]['k3'])*(1/4.184)*(1+Cos(int(dicts_improper[i]['periodicity3'])* phi_im - float(dicts_improper[i]['phase3'])))
                imp4= float(dicts_improper[i]['k4'])*(1/4.184)*(1+Cos(int(dicts_improper[i]['periodicity4'])* phi_im - float(dicts_improper[i]['phase4'])))
                imp_fun = imp1 + imp2 +imp3 +imp4
                improperfuncs.set(improper_id, imp_fun)
                #print(improperfuncs.potentials())

                for t in range(1,5):
                    mol_params.add(improper_id, float(dicts_improper[i]['k%s'%t])*(1/4.184), int(dicts_improper[i]['periodicity%s'%t]), float(dicts_improper[i]['phase%s'%t]) ) 


            mol = editmol.setProperty("bond", bondfuncs).commit()
            mol = editmol.setProperty("angle" , anglefuncs).commit()
            mol = editmol.setProperty("dihedral" , dihedralfuncs).commit()
            mol = editmol.setProperty("improper" , improperfuncs).commit()
            system.update(mol)

        # Now we work out non bonded pairs see SireIO/amber.cpp L2213


            print("Set up nbpairs")
            print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")
            ## Define the bonded pairs in a list that is called are12
            #print("Now calculating 1-2 intercactions")
            are12 = []
            for i in range(0, natoms): 
                for j in range (0, natoms): 
                    if conn.areBonded(atoms[i].index(), atoms[j].index()) == True:
                        #ij = {}
                        ij= (i,j)
                        are12.append(ij)
            are12_bckup = are12[:]


            #print("Now calculating 1-3 intercactions")
            are13 = []
            for i in range(0, natoms): 
                for j in range (0, natoms): 
                    if conn.areAngled(atoms[i].index(), atoms[j].index()) == True:
                        ij = {}
                        ij= (i,j)
                        are13.append(ij)
            are13_bckup = are13[:]

           # print("Now calculating 1-4 intercactions")
            are14 = []
            for i in range(0, natoms): 
                for j in range (0, natoms):
                   
                    if conn.areDihedraled(atoms[i].index(), atoms[j].index()) == True and conn.areAngled(atoms[i].index(), atoms[j].index()) == False:
                        ij = {}
                        ij= (i,j)
                        are14.append(ij)
            are14_bckup = are14[:]

           # print("Now calculating the non-bonded intercactions")
            bonded_pairs_list = are12_bckup + are13_bckup + are14_bckup    
            nb_pair_list =[]

            for i in range(0, natoms): 
                #print("i=",i)
                for j in range (0, natoms):
                    if i != j and (i,j) not in bonded_pairs_list:
                        nb_pair_list.append((i,j))
            are_nb_bckup = nb_pair_list[:]

            nbpairs = CLJNBPairs(editmol.info(), CLJScaleFactor(0,0))
            #print("Now setting 1-2 intercactions")
            for i in range(0, len(are12)):
                scale_factor1 = 0
                scale_factor2 = 0
                nbpairs.set(atoms.index( int(are12[i][0])), atoms.index(int(are12[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))

            #print("Now setting 1-3 intercactions")
            for i in range(0, len(are13)):
                scale_factor1 = 0
                scale_factor2 = 0
                nbpairs.set(atoms.index( int(are13[i][0])), atoms.index(int(are13[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))

           # print("Now setting 1-4 intercactions")              
            for i in range(0, len(are14)): 
                scale_factor1 = 1/2
                scale_factor2 = 1/2
                nbpairs.set(atoms.index( int(are14[i][0])), atoms.index(int(are14[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                mol_params.add14Pair(BondID(atoms.index( int(are14[i][0])), atoms.index( int(are14[i][1]))),scale_factor1 , scale_factor2)

           # print("Now setting non-bonded intercactions")  
            #print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")              
            for i in range(0, len(nb_pair_list)):  
                scale_factor1 = 1
                scale_factor2 = 1
                nbpairs.set(atoms.index( int(nb_pair_list[i][0])), atoms.index(int(nb_pair_list[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))

                # print("~~~~~~~~~~~~~~~~~~`")
                
            mol = editmol.setProperty("intrascale" , nbpairs).commit()
            system.update(mol)


        #print("Setup name of qube FF")
        # mol = mol.edit().setProperty("forcefield", ffToProperty("qube")).commit()
        # system.update(mol)

        molecule = editmol.commit()
        newmolecules.add(molecule)



    return (newmolecules, space)

def assignVirtualSites(system, xmlfile):
    import xml.dom.minidom as minidom
    xmldoc = minidom.parse(xmlfile)

    #Get the virtual sites from the xml file 
    itemlist_atom = xmldoc.getElementsByTagName('Atom')
    dicts_atom = []
    for items in itemlist_atom:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_atom.append(d)
    dicts_at =  str(dicts_atom).split()


    itemlist_VirtualSite = xmldoc.getElementsByTagName('VirtualSite')
    dicts_virtualsite = []
    for items in itemlist_VirtualSite:
        d = {}
        for a in items.attributes.values():
            d[a.name] = a.value
        dicts_virtualsite.append(d)
    nVirtualSites = itemlist_VirtualSite.length

    v_site_CLJ = []
    for i in range(0, int(len(dicts_atom))):
        if dicts_atom[i]['type'][0] == 'v':
            v_site_CLJ = dicts_atom[i]
            dicts_virtualsite.append(v_site_CLJ)

    for i in range(0, len(itemlist_VirtualSite)):
        dicts_virtualsite[i].update(dicts_virtualsite[i+len(itemlist_VirtualSite)])
        dicts_virtualsite[i].update(dicts_virtualsite[i+2*len(itemlist_VirtualSite)])
    dict_vs = []
    for i in range(0, len(itemlist_VirtualSite)):
        dicts_virtualsite[i]
        dict_vs.append(dicts_virtualsite[i])


    molnums = molecules.molNums()
    for molnum in molnums:
        mol = molecules.molecule(molnum)[0].molecule()
        mol = mol.edit().setProperty("virtual-sites", vsiteListToProperty(dict_vs)).commit()
       # xml_mol = xml_molecules.molecule(molnum)[0].molecule()
        system.update(mol)

    # molecules = system.molecules()

    return(system)



def getDummies(molecule):
    print ("Selecting dummy groups")
    natoms = molecule.nAtoms()
    atoms = molecule.atoms()

    from_dummies = None
    to_dummies = None

    for x in range(0, natoms):
        atom = atoms[x]
        if atom.property("initial_ambertype") == "du":
            if from_dummies is None:
                from_dummies = molecule.selectAll(atom.index())
            else:
                from_dummies += molecule.selectAll(atom.index())
        elif atom.property("final_ambertype") == "du":
            if to_dummies is None:
                to_dummies = molecule.selectAll(atom.index())
            else:
                to_dummies += molecule.selectAll(atom.index())

    return to_dummies, from_dummies

#molecules = ["methylformate", "acetamide", "N-methylacetamide", "dimethylacetamide", "nitroethane", "benzene", "benzonitrile"]
#molecules = ["toluene", "nitrobenzene", "acetophenone", "benzamide", "phenol", "chlorobenzene"]
#molecules = ["methanol", "N-methylaniline", "fluorobenzene", "trifluoromethylbenzene", "bromobenzene", "1,3-dichloropropane"]
#molecules = ["formaldehyde", "2-hexanone" , "2-heptanone", "cyclopentanone", "cyclohexanone", "ethyl_acetate", "pentanenitrile" , "propionitrile",  "1-propanamine", "1-butanamine"]
#molecules = ["nitro-methane","1-nitropropane" ,"2-nitropropane" ,"N,N-dimethylformamide" ,"ethanol" ,"1-butanol" ,"2-methylpropan-2-ol", "1-pentanol", "3-pentanol" ,"2-methyl-2-butanol" ,"1-octanol"]
#molecules=[ "nitroethane", "2-hexanone","ethyl_acetate", "styrene", "benzaldehyde"] 
#molecules =[ "ethylbenzene", "1,2-dimethylbenzene", "1-methylethylbenene","1,2,4-trimethylbenzene" ,
#"styrene" , "benzaldehyde" ,"o-chloroaniline" ,"2-methylphenol" ,"4-methylphenol", "benzyl_alcohol"] 
#molecules = ["propane", "propene", "acetone", "acetaldehyde"]
#molecules = ["acetic_acid", "methylacetate", "acetamide", "nitroethane"]
molecules = [ "methylformate"]

results = []
for mols in molecules:
    res ={}
os.chdir("/home/sofia/Desktop/Sofia/SI_data/molecules/"+mols)
xmlfile1 = "MOL_extra.xml"
pdbfile1 = "MOL.pdb"
(molecules, space) = readXmlParameters(pdbfile1, xmlfile1)
system1 = createSystemFreeEnergy(molecules)
system1 = setupForceFieldsFreeEnergy(system1, space)

ranseed = RanGenerator().randInt(100000, 1000000)
moves = setupMovesFreeEnergy(system1, ranseed, gpu.val, lambda_val)
mdmoves = moves.moves()[0]

integrator = mdmoves.integrator()
nrg_somd = integrator.getPotentialEnergy(system1)

print("The SOMD energy is:",  nrg_somd)

nrg_sire = system1.energy()
print ("nrg_somd is %s" % nrg_somd)
print ("nrg_sire is %s" % nrg_sire)
diff = nrg_sire - nrg_somd

print ("@@@@ The single point energy difference between sire and somd is %s " % (diff) )

# TODO check components to work out discrepancy
# make test script work with various config settings (no cutoff/cutoffs)
# print ("Detailed breakdown of Sire energies")
# energyComponents = system1.energies()
# for component in energyComponents.keys():
#     print ("Component %s energy %s " % (component, system1.energy(component)))


res = (mols, nrg_somd)
results.append(res)

results


    
