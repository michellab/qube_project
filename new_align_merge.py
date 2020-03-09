

import tempfile as _tempfile
import os as _os
import subprocess as _subprocess
import tempfile as _tempfile

import warnings as _warnings
# Suppress numpy warnings from RDKit import.
_warnings.filterwarnings("ignore", message="numpy.dtype size changed")
_warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# Suppress duplicate to-Python converted warnings.
# Both Sire and RDKit register the same converter.
with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore")
    from rdkit import Chem as _Chem
    from rdkit.Chem import rdFMCS as _rdFMCS

from Sire import Base as _SireBase
from Sire import Maths as _SireMaths
from Sire import Mol as _SireMol
from Sire import Units as _SireUnits
from BioSimSpace._Exceptions import AlignmentError as _AlignmentError
from BioSimSpace._Exceptions import MissingSoftwareError as _MissingSoftwareError
from BioSimSpace._SireWrappers import Molecule as _Molecule

from BioSimSpace import IO as _IO
from BioSimSpace import Units as _Units
from BioSimSpace import _Utils as _Utils

import os
import re
import sys
import argparse
from Sire.Base import *
from datetime import datetime
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

#########################################
#       Config file parameters          #  
#########################################



combining_rules = Parameter("combining rules", "geometric",
                            """Combining rules to use for the non-bonded interactions.""")

cutoff_type = Parameter("cutoff type", "nocutoff", """The cutoff method to use during the simulation.""")

cutoff_dist = Parameter("cutoff distance", 500 * angstrom,
                        """The cutoff distance to use for the non-bonded interactions.""")

use_restraints = Parameter("use restraints", False, """Whether or not to use harmonic restaints on the solute atoms.""")


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


def vsiteListToProperty(list):
    prop = Properties()
    i = 0
    for entry in list:
        for key, value in entry.items():
            prop.setProperty("%s(%d)" % (key,i), VariantProperty(value))
        i += 1
    prop.setProperty("nvirtualsites",VariantProperty(i))
    return prop



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
    #print (dicts_angle)

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
                else:#if QUBE_
                    a1 = int(to_str1.replace("[","").replace("]","").replace("'","") )
              
                at1.append(a1)
          

            at2 = []
            for i in range(0, nAngles):
                a2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_angle[i]['class2'])))
                if dicts_atom[i]['type'][0] == 'o': #if opls_
                    a2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                else: #if QUBE_

                    a2 = int(to_str2.replace("[","").replace("]","").replace("'","") )
                
                at2.append(a2)
         

            at3 = []
            for i in range(0, nAngles):
                a3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_angle[i]['class3'])))
                if dicts_atom[i]['type'][0] == 'o': #if opls_
                    a3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800
                else: #if QUBE_
                    a3 = int(to_str3.replace("[","").replace("]","").replace("'","") )
               
                at3.append(a3)

            theta = internalff.symbols().angle().theta()
            for j in range(0,nAngles):
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
                else:  #if QUBE_
                    d1 = int(to_str1.replace("[","").replace("]","").replace("'","") )
               
                di1.append(d1)


            di2 = []
            for i in range(0, nProper):
                d2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_proper[i]['class2'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                else: #if QUBE_
                    d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )
                
                di2.append(d2)

            di3 = []
            for i in range(0, nProper):
                d3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_proper[i]['class3'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800
                else: #if QUBE_
                    d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )
                
                di3.append(d3)
          

            di4 = []
            for i in range(0, nProper):
                d4 = {}
                to_str4 = str(re.findall(r"\d+",str(dicts_proper[i]['class4'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )-800
                else: #if QUBE_
                    d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )
                
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
                else:
                    d1 = int(to_str1.replace("[","").replace("]","").replace("'","") )
                
                di_im1.append(d1)


            di_im2 = []
            for i in range(0, nImproper):
                d2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_improper[i]['class2'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                else:
                    d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )

                
                di_im2.append(d2)

            di_im3 = []
            for i in range(0, nImproper):
                d3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_improper[i]['class3'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800
                else:
                    d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )
                
                di_im3.append(d3)

            di_im4 = []
            for i in range(0, nImproper):
                d4 = {}
                to_str4 = str(re.findall(r"\d+",str(dicts_improper[i]['class4'])))
                if dicts_atom[0]['type'][0] == 'o':#if opls_
                    d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )-800
                else:
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
        mol = mol.edit().setProperty("forcefield", ffToProperty("qube")).commit()
        system.update(mol)

        molecule = editmol.commit()
        newmolecules.add(molecule)



    return (newmolecules, space)


def ffToProperty(string):
    prop = Properties()      
    prop.setProperty("forcefield",VariantProperty("qube"))
    return prop



xmlfile = "G1_4a.xml" 
pdbfile = "G1_4a.pdb" 
(molecules, space) = readXmlParameters(pdbfile, xmlfile) 
 
 
xmlfile1 = "G1_4f.xml" 
pdbfile1 = "G1_4f.pdb" 
(molecules1, space1) = readXmlParameters(pdbfile1, xmlfile1) 

mol1 = molecules.molecule(MolNum(1))
mol2 = molecules1.molecule(MolNum(2))

prematch = {}
import BioSimSpace as BSS
bss_mol1 = BSS.IO.readMolecules(pdbfile)
bss_mol2 = BSS.IO.readMolecules(pdbfile1)

def _score_rdkit_mappings(molecule0, molecule1, rdkit_molecule0, rdkit_molecule1,
        mcs_smarts, prematch, scoring_function, property_map0, property_map1):
    """Internal function to score atom mappings based on the root mean squared
       displacement (RMSD) between mapped atoms in two molecules. Optionally,
       molecule0 can first be aligned to molecule1 based on the mapping prior
       to computing the RMSD. The function returns the mappings sorted based
       on their score from best to worst, along with a list containing the
       scores for each mapping.
       Parameters
       ----------
       molecule0 : Sire.Molecule.Molecule
           The first molecule (Sire representation).
       molecule0 : Sire.Molecule.Molecule
           The second molecule (Sire representation).
       rdkit_mol0 : RDKit.Chem.Mol
           The first molecule (RDKit representation).
       rdkit_mol1 : RDKit.Chem.Mol
           The second molecule (RDKit representation).
       mcs_smarts : RDKit.Chem.MolFromSmarts
           The smarts string representing the maximum common substructure of
           the two molecules.
       prematch : dict
           A dictionary of atom mappings that must be included in the match.
       scoring_function : str
           The RMSD scoring function.
       property_map0 : dict
           A dictionary that maps "properties" in molecule0 to their user
           defined values. This allows the user to refer to properties
           with their own naming scheme, e.g. { "charge" : "my-charge" }
       property_map1 : dict
           A dictionary that maps "properties" in molecule1 to their user
           defined values.
       Returns
       -------
       mapping, scores : ([dict], list)
           The ranked mappings and corresponding scores.
    """

    # Adapted from FESetup: https://github.com/CCPBioSim/fesetup

    # Make sure to re-map the coordinates property in both molecules, otherwise
    # the move and align functions from Sire will not work.
    prop0 = property_map0.get("coordinates", "coordinates")
    prop1 = property_map1.get("coordinates", "coordinates")

    if prop0 != "coordinates":
        molecule0 = molecule0.edit().setProperty("coordinates", molecule0.property(prop0)).commit()
    if prop1 != "coordinates":
        molecule1 = molecule1.edit().setProperty("coordinates", molecule1.property(prop1)).commit()

    # Get the set of matching substructures in each molecule. For some reason
    # setting uniquify to True removes valid matches, in some cases even the
    # best match! As such, we set uniquify to False and account ignore duplicate
    # mappings in the code below.
    matches0 = rdkit_molecule0.GetSubstructMatches(mcs_smarts, uniquify=False, maxMatches=1000, useChirality=False)
    matches1 = rdkit_molecule1.GetSubstructMatches(mcs_smarts, uniquify=False, maxMatches=1000, useChirality=False)

    # Swap the order of the matches.
    if len(matches0) < len(matches1):
        matches0, matches1 = matches1, matches0
        is_swapped = True
    else:
        is_swapped = False

    # Initialise a list to hold the mappings.
    mappings = []

    # Initialise a list of to hold the score for each mapping.
    scores = []

    # Loop over all matches from mol0.
    for x in range(len(matches0)):
        match0 = matches0[x]

        # Loop over all matches from mol1.
        for y in range(len(matches1)):
            match1 = matches1[y]

            # Initialise the mapping for this match.
            mapping = {}
            sire_mapping = {}

            # Loop over all atoms in the match.
            for i, idx0 in enumerate(match0):
                idx1 = match1[i]

                # Add to the mapping.
                if is_swapped:
                    mapping[idx1] = idx0
                    sire_mapping[_SireMol.AtomIdx(idx1)] = _SireMol.AtomIdx(idx0)
                else:
                    mapping[idx0] = idx1
                    sire_mapping[_SireMol.AtomIdx(idx0)] = _SireMol.AtomIdx(idx1)

            # This is a new mapping:
            if not mapping in mappings:
                # Check that the mapping contains the pre-match.
                is_valid = True
                for idx0, idx1 in prematch.items():
                    # Pre-match isn't found, return to top of loop.
                    if idx0 not in mapping or mapping[idx0] != idx1:
                        is_valid = False
                        break

                if is_valid:
                    # Rigidly align molecule0 to molecule1 based on the mapping.
                    if scoring_function == "RMSDALIGN":
                        try:
                            molecule0 = molecule0.move().align(molecule1, _SireMol.AtomResultMatcher(sire_mapping)).molecule()
                        except Exception as e:
                            msg = "Failed to align molecules when scoring based on mapping: %r" % mapping
                            if _isVerbose():
                                raise _AlignmentError(msg) from e
                            else:
                                raise _AlignmentError(msg) from None
                    # Flexibly align molecule0 to molecule1 based on the mapping.
                    elif scoring_function == "RMSDFLEXALIGN":
                        molecule0 = flexAlign(_Molecule(molecule0), _Molecule(molecule1), mapping,
                            property_map0=property_map0, property_map1=property_map1)._sire_object

                    # Append the mapping to the list.
                    mappings.append(mapping)

                    # We now compute the RMSD between the coordinates of the matched atoms
                    # in molecule0 and molecule1.

                    # Initialise lists to hold the coordinates.
                    c0 = []
                    c1 = []

                    # Loop over each atom index in the map.
                    for idx0, idx1 in sire_mapping.items():
                        # Append the coordinates of the matched atom in molecule0.
                        c0.append(molecule0.atom(idx0).property("coordinates"))
                        # Append the coordinates of atom in molecule1 to which it maps.
                        c1.append(molecule1.atom(idx1).property("coordinates"))

                    # Compute the RMSD between the two sets of coordinates.
                    scores.append(_SireMaths.getRMSD(c0, c1))

    # No mappings were found.
    if len(mappings) == 0:
        if len(prematch) == 0:
            return ([{}], [])
        else:
            return ([prematch], [])

    # Sort the scores and return the sorted keys. (Smaller RMSD is best)
    keys = sorted(range(len(scores)), key=lambda k: scores[k])

    # Sort the mappings.
    mappings = [mappings[x] for x in keys]

    # Sort the scores and convert to Angstroms.
    scores = [scores[x] * _Units.Length.angstrom for x in keys]

    # Return the sorted mappings and their scores.
    return (mappings, scores)
def _validate_mapping(molecule0, molecule1, mapping, name):
    """Internal function to validate that a mapping contains key:value pairs
       of the correct type.
       Parameters
       ----------
       molecule0 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The molecule of interest.
       molecule1 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The reference molecule.
       mapping : dict
           The mapping between matching atom indices in the two molecules.
       name : str
           The name of the mapping. (Used when raising exceptions.)
    """

    for idx0, idx1 in mapping.items():
            if type(idx0) is int and type(idx1) is int:
                pass
            elif type(idx0) is _SireMol.AtomIdx and type(idx1) is _SireMol.AtomIdx:
                idx0 = idx0.value()
                idx1 = idx1.value()
            else:
                raise TypeError("%r dictionary key:value pairs must be of type 'int' or "
                                "'Sire.Mol.AtomIdx'" % name)
            if idx0 < 0 or idx0 >= molecule0.nAtoms() or \
               idx1 < 0 or idx1 >= molecule1.nAtoms():
                raise ValueError("%r dictionary key:value pair '%s : %s' is out of range! "
                                 "The molecules contain %d and %d atoms."
                                 % (name, idx0, idx1, molecule0.nAtoms(), molecule1.nAtoms()))


def matchAtoms(molecule0,
               molecule1,
               scoring_function="rmsd_align",
               matches=1,
               return_scores=False,
               prematch={},
               timeout=5*_Units.Time.second,
               property_map0={},
               property_map1={}):
    """Find mappings between atom indices in molecule0 to those in molecule1.
       Molecules are aligned using a Maximum Common Substructure (MCS) search.
       When requesting more than one match, the mappings will be sorted using
       a scoring function and returned in order of best to worst score. (Note
       that, depending on the scoring function the "best" score may have the
       lowest value.)
       Parameters
       ----------
       molecule0 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The molecule of interest.
       molecule1 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The reference molecule.
       scoring_function : str
           The scoring function used to match atoms. Available options are:
             - "rmsd"
                 Calculate the root mean squared distance between the
                 coordinates of atoms in molecule0 to those that they
                 map to in molecule1.
             - "rmsd_align"
                 Align molecule0 to molecule1 based on the mapping before
                 computing the above RMSD score.
             - "rmsd_flex_align"
                 Flexibly align molecule0 to molecule1 based on the mapping
                 before computing the above RMSD score. (Requires the
                 'fkcombu'. package: http://strcomp.protein.osaka-u.ac.jp/kcombu)
       matches : int
           The maximum number of matches to return. (Sorted in order of score).
       return_scores : bool
           Whether to return a list containing the scores for each mapping.
       prematch : dict
           A dictionary of atom mappings that must be included in the match.
       timeout : BioSimSpace.Types.Time
           The timeout for the maximum common substructure search.
       property_map0 : dict
           A dictionary that maps "properties" in molecule0 to their user
           defined values. This allows the user to refer to properties
           with their own naming scheme, e.g. { "charge" : "my-charge" }
       property_map1 : dict
           A dictionary that maps "properties" in molecule1 to their user
           defined values.
       Returns
       -------
       matches : dict, [dict], ([dict], list)
           The best atom mapping, a list containing a user specified number of
           the best mappings ranked by their score, or a tuple containing the
           list of best mappings and a list of the corresponding scores.
       Examples
       --------
       Find the best maximum common substructure mapping between two molecules.
       >>> import BioSimSpace as BSS
       >>> mapping = BSS.Align.matchAtoms(molecule0, molecule1)
       Find the 5 best mappings.
       >>> import BioSimSpace as BSS
       >>> mappings = BSS.Align.matchAtoms(molecule0, molecule1, matches=5)
       Find the 5 best mappings along with their ranking scores.
       >>> import BioSimSpace as BSS
       >>> mappings, scores = BSS.Align.matchAtoms(molecule0, molecule1, matches=5, return_scores=True)
       Find the 5 best mappings along with their ranking scores. Score
       by flexibly aligning molecule0 to molecule1 based on each mapping
       and computing the root mean squared displacement of the matched
       atoms.
       >>> import BioSimSpace as BSS
       >>> mappings, scores = BSS.Align.matchAtoms(molecule0, molecule1, matches=5, return_scores=True, scoring_function="rmsd_flex_align")
       Find the best mapping that contains a prematch (this is a dictionary mapping
       atom indices in molecule0 to those in molecule1).
       >>> import BioSimSpace as BSS
       >>> mapping = BSS.Align.matchAtoms(molecule0, molecule1, prematch={0 : 10, 3 : 7})
    """

    # A list of supported scoring functions.
    scoring_functions = ["RMSD", "RMSDALIGN", "RMSDFLEXALIGN"]

    # Validate input.

    if type(molecule0) is not Molecule:
        raise TypeError("'molecule0' must be of type 'Sire.Mol._Mol.Molecule'")

    if type(molecule1) is not Molecule:
        raise TypeError("'molecule1' must be of type Sire.Mol._Mol.Molecule'")

    if type(scoring_function) is not str:
        raise TypeError("'scoring_function' must be of type 'str'")
    else:
        # Strip underscores and whitespace, then convert to upper case.
        _scoring_function = scoring_function.replace("_", "").upper()
        _scoring_function = _scoring_function.replace(" ", "").upper()
        if not _scoring_function in scoring_functions:
            raise ValueError("Unsupported scoring function '%s'. Options are: %s"
                % (scoring_function, scoring_functions))

    if _scoring_function == "RMSDFLEXALIGN" and _fkcombu_exe is None:
        raise _MissingSoftwareError("'rmsd_flex_align' option requires the 'fkcombu' program: "
                                    "http://strcomp.protein.osaka-u.ac.jp/kcombu")

    if type(matches) is not int:
        raise TypeError("'matches' must be of type 'int'")
    else:
        if matches < 0:
            raise ValueError("'matches' must be positive!")

    if type(return_scores) is not bool:
        raise TypeError("'return_matches' must be of type 'bool'")

    if type(prematch) is not dict:
        raise TypeError("'prematch' must be of type 'dict'")
    else:
        _validate_mapping(molecule0, molecule1, prematch, "prematch")

    # if type(timeout) is not _Units.Time._Time:
    #     raise TypeError("'timeout' must be of type 'BioSimSpace.Types.Time'")

    if type(property_map0) is not dict:
        raise TypeError("'property_map0' must be of type 'dict'")

    if type(property_map1) is not dict:
        raise TypeError("'property_map1' must be of type 'dict'")

    # Extract the Sire molecule from each BioSimSpace molecule.
    # mol0 = molecule0._getSireObject()
    # mol1 = molecule1._getSireObject()

    # Convert the timeout to seconds and take the magnitude as an integer.
    timeout = int(timeout.seconds().magnitude())

    # Create a temporary working directory.
    tmp_dir = _tempfile.TemporaryDirectory()
    work_dir = tmp_dir.name

    # Use RDKkit to find the maximum common substructure.

    try:
        # Run inside a temporary directory.
        with _Utils.cd(work_dir):
            # Write both molecules to PDB files.
            _IO.saveMolecules("tmp0", bss_mol1, "PDB", property_map=property_map0)
            _IO.saveMolecules("tmp1", bss_mol2, "PDB", property_map=property_map1)

            # Load the molecules with RDKit.
            # Note that the C++ function overloading seems to be broken, so we
            # need to pass all arguments by position, rather than keyword.
            # The arguments are: "filename", "sanitize", "removeHs", "flavor"
            mols = [_Chem.MolFromPDBFile("tmp0.pdb", False, False, 0),
                    _Chem.MolFromPDBFile("tmp1.pdb", False, False, 0)]

            # Generate the MCS match.
            mcs = _rdFMCS.FindMCS(mols, atomCompare=_rdFMCS.AtomCompare.CompareAny,
                bondCompare=_rdFMCS.BondCompare.CompareAny, completeRingsOnly=True,
                ringMatchesRingOnly=True, matchChiralTag=False, matchValences=False,
                maximizeBonds=False, timeout=timeout)

            # Get the common substructure as a SMARTS string.
            mcs_smarts = _Chem.MolFromSmarts(mcs.smartsString)

    except:
        raise RuntimeError("RDKIT MCS mapping failed!")

    # Score the mappings and return them in sorted order (best to worst).
    mappings, scores = _score_rdkit_mappings(mol1, mol2, mols[0], mols[1],
        mcs_smarts, prematch, _scoring_function, property_map0, property_map1)

    # Sometimes RDKit fails to generate a mapping that includes the prematch.
    # If so, then try generating a mapping using the MCS routine from Sire.
    if len(mappings) == 1 and mappings[0] == prematch:

        # Convert timeout to a Sire Unit.
        timeout = timeout * _SireUnits.second

        # Regular match. Include light atoms, but don't allow matches between heavy
        # and light atoms.
        m0 = mol0.evaluate().findMCSmatches(mol1, _SireMol.AtomResultMatcher(_to_sire_mapping(prematch)),
                                            timeout, True, property_map0, property_map1, 6, False)

        # Include light atoms, and allow matches between heavy and light atoms.
        # This captures mappings such as O --> H in methane to methanol.
        m1 = mol0.evaluate().findMCSmatches(mol1, _SireMol.AtomResultMatcher(_to_sire_mapping(prematch)),
                                            timeout, True, property_map0, property_map1, 0, False)

        # Take the mapping with the larger number of matches.
        if len(m1) > 0:
            if len(m0) > 0:
                if len(m1[0]) > len(m0[0]):
                    mappings = m1
                else:
                    mappings = m0
            else:
                mappings = m1
        else:
            mappings = m0

        # Score the mappings and return them in sorted order (best to worst).
        mappings, scores = _score_sire_mappings(mol0, mol1, mappings, prematch,
            _scoring_function, property_map0, property_map1)

    if matches == 1:
        if return_scores:
            return (mappings[0], scores[0])
        else:
            return mappings[0]
    else:
        # Return a list of matches from best to worst.
        if return_scores:
            return (mappings[0:matches], scores[0:matches])
        # Return a tuple containing the list of matches from best to
        # worst along with the list of scores.
        else:
            return mappings[0:matches]


mappings = matchAtoms(mol1, mol2)

inverted_mapping = dict([[v,k] for k,v in mappings.items()])


def _to_sire_mapping(mapping):
    """Internal function to convert a regular mapping to Sire AtomIdx format.
       Parameters
       ----------
       mapping : {int:int}
           The regular mapping.
       Returns
       -------
       sire_mapping : {Sire.Mol.AtomIdx:Sire.Mol.AtomIdx}
           The Sire mapping.
    """

    sire_mapping = {}

    # Convert the mapping to AtomIdx key:value pairs.
    for idx0, idx1 in mapping.items():
        # Early exit if the mapping is already the correct format.
        if type(idx0) is _SireMol.AtomIdx:
            return mapping
        else:
            sire_mapping[_SireMol.AtomIdx(idx0)] = _SireMol.AtomIdx(idx1)

    return sire_mapping


def rmsdAlign(molecule0, molecule1, mapping=None, property_map0={}, property_map1={}):
    """Align atoms in molecule0 to those in molecule1 using the mapping
       between matched atom indices. The molecule is aligned using rigid-body
       translation and rotations, with a root mean squared displacement (RMSD)
       fit to find the optimal translation vector (as opposed to merely taking
       the difference of centroids).
       Parameters
       ----------
       molecule0 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The molecule to align.
       molecule1 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The reference molecule.
       mapping : dict
           A dictionary mapping atoms in molecule0 to those in molecule1.
       property_map0 : dict
           A dictionary that maps "properties" in molecule0 to their user
           defined values. This allows the user to refer to properties
           with their own naming scheme, e.g. { "charge" : "my-charge" }
       property_map1 : dict
           A dictionary that maps "properties" in molecule1 to their user
           defined values.
       Returns
       -------
       molecule : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The aligned molecule.
       Examples
       --------
       Align molecule0 to molecule1 based on a precomputed mapping.
       >>> import BioSimSpace as BSS
       >>> molecule0 = BSS.Align.rmsdAlign(molecule0, molecule1, mapping)
       Align molecule0 to molecule1. Since no mapping is passed one will be
       autogenerated using :class:`matchAtoms <BioSimSpace.Align.matchAtoms>`
       with default options.
       >>> import BioSimSpace as BSS
       >>> molecule0 = BSS.Align.rmsdAlign(molecule0, molecule1)
    """

    # if type(molecule0) is not Molecule:
    #     raise TypeError("'molecule0' must be of type 'BioSimSpace._SireWrappers.Molecule'")

    # if type(molecule1) is not Molecule:
    #     raise TypeError("'molecule1' must be of type 'BioSimSpace._SireWrappers.Molecule'")

    if type(property_map0) is not dict:
        raise TypeError("'property_map0' must be of type 'dict'")

    if type(property_map1) is not dict:
        raise TypeError("'property_map1' must be of type 'dict'")

    # The user has passed an atom mapping.
    if mapping is not None:
        if type(mapping) is not dict:
            raise TypeError("'mapping' must be of type 'dict'.")
        else:
            _validate_mapping(molecule0, molecule1, mapping, "mapping")

    # Get the best match atom mapping.
    else:
        mapping = matchAtoms(molecule0, molecule1, property_map0=property_map0,
                             property_map1=property_map1)

    # Extract the Sire molecule from each BioSimSpace molecule.
    # mol0 = molecule0._getSireObject()
    # mol1 = molecule1._getSireObject()

    # Convert the mapping to AtomIdx key:value pairs.
    sire_mapping = _to_sire_mapping(mapping)

    # Perform the alignment, mol0 to mol1.
    try:
        molecule0 = molecule0.move().align(molecule1, _SireMol.AtomResultMatcher(sire_mapping)).molecule()
    except Exception as e:
        msg = "Failed to align molecules based on mapping: %r" % mapping
        # if _isVerbose():
        #     raise _AlignmentError(msg) from e
        # else:
        #     raise _AlignmentError(msg) from None

    # Return the aligned molecule.
    return _Molecule(molecule1)


lig1 = mol1
lig2 = rmsdAlign(lig1, mol2, mappings)  



def sire_map(molecule0, molecule1, mapping=None, allow_ring_breaking=False,
          property_map0={}, property_map1={}):
    """Create a merged molecule from 'molecule0' and 'molecule1' based on the
       atom index 'mapping'. The merged molecule can be used in single- and
       dual-toplogy free energy calculations.
       Parameters
       ----------
       molecule0 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           A molecule object.
       molecule1 : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           A second molecule object.
       mapping : dict
           The mapping between matching atom indices in the two molecules.
           If no mapping is provided, then atoms in molecule0 will be mapped
           to those in molecule1 using "matchAtoms", with "rmsdAlign" then
           used to align molecule0 to molecule1 based on the resulting mapping.
       allow_ring_breaking : bool
           Whether to allow the opening/closing of rings during a merge.
       allow_ring_size_change : bool
           Whether to allow changes in ring size.
       property_map0 : dict
           A dictionary that maps "properties" in molecule0 to their user
           defined values. This allows the user to refer to properties
           with their own naming scheme, e.g. { "charge" : "my-charge" }
       property_map1 : dict
           A dictionary that maps "properties" in molecule1 to their user
           defined values.
       Returns
       -------
       molecule : :class:`Molecule <BioSimSpace._SireWrappers.Molecule>`
           The merged molecule.
       Examples
       --------
       Merge molecule0 and molecule1 based on a precomputed mapping.
       >>> import BioSimSpace as BSS
       >>> merged = BSS.Align.merge(molecule0, molecule1, mapping)
       Merge molecule0 with molecule1. Since no mapping is passed one will be
       autogenerated using :class:`matchAtoms <BioSimSpace.Align.matchAtoms>`
       with default options, following which :class:`rmsdAlign <BioSimSpace.Align.rmsdAlign>`
       will be used to align molecule0 to molecule1 based on the resulting mapping.
       >>> import BioSimSpace as BSS
       >>> molecule0 = BSS.Align.merge(molecule0, molecule1)
    """

    # if type(molecule0) is not _Molecule:
    #     raise TypeError("'molecule0' must be of type 'BioSimSpace._SireWrappers.Molecule'")

    # if type(molecule1) is not _Molecule:
    #     raise TypeError("'molecule1' must be of type 'BioSimSpace._SireWrappers.Molecule'")

    if type(property_map0) is not dict:
        raise TypeError("'property_map0' must be of type 'dict'")

    if type(property_map1) is not dict:
        raise TypeError("'property_map1' must be of type 'dict'")

    if type(allow_ring_breaking) is not bool:
        raise TypeError("'allow_ring_breaking' must be of type 'bool'")

    # if type(allow_ring_size_change) is not bool:
    #     raise TypeError("'allow_ring_size_change' must be of type 'bool'")

    # The user has passed an atom mapping.
    if mapping is not None:
        if type(mapping) is not dict:
            raise TypeError("'mapping' must be of type 'dict'.")
        else:
            _validate_mapping(molecule0, molecule1, mapping, "mapping")
            molecule0 = _Molecule(lig1)

    # Get the best atom mapping and align molecule0 to molecule1 based on the
    # mapping.
    else:
        mapping = matchAtoms(molecule0, molecule1, property_map0=property_map0, property_map1=property_map1)
        molecule0 = rmsdAlign(molecule0, molecule1, mapping)

    # Convert the mapping to AtomIdx key:value pairs.
    sire_mapping = _to_sire_mapping(mapping)

    # Create and return the merged molecule.
    return molecule0, molecule1, sire_mapping


    # return molecule0._merge(molecule1, sire_mapping, allow_ring_breaking=allow_ring_breaking,
    #          property_map0=property_map0, property_map1=property_map1)

( molecule0, molecule1, sire_mapping)= sire_map(lig1, lig2, mappings)

from pytest import approx as _approx

import os.path as _path
import random as _random
import string as _string

from Sire import Base as _SireBase
from Sire import CAS as _SireCAS
from Sire import MM as _SireMM
from Sire import Mol as _SireMol
from Sire import Units as _SireUnits


from BioSimSpace._Exceptions import IncompatibleError as _IncompatibleError
from BioSimSpace.Types import Length as _Length



def _is_ring_broken(conn0, conn1, idx0, idy0, idx1, idy1):
    """Internal function to test whether a perturbation changes the connectivity
       around two atoms such that a ring is broken.
       Parameters
       ----------
       conn0 : Sire.Mol.Connectivity
           The connectivity object for the first end state.
       conn1 : Sire.Mol.Connectivity
           The connectivity object for the second end state.
       idx0 : Sire.Mol.AtomIdx
           The index of the first atom in the first state.
       idy0 : Sire.Mol.AtomIdx
           The index of the second atom in the first state.
       idx1 : Sire.Mol.AtomIdx
           The index of the first atom in the second state.
       idy1 : Sire.Mol.AtomIdx
           The index of the second atom in the second state.
    """

    # Have we opened/closed a ring? This means that both atoms are part of a
    # ring in one end state (either in it, or on it), whereas at least one
    # are the result of changes in ring size, where atoms remain in or on a
    # ring in both end states.

    # Whether each atom is in a ring in both end states.
    in_ring_idx0 = conn0.inRing(idx0)
    in_ring_idy0 = conn0.inRing(idy0)
    in_ring_idx1 = conn1.inRing(idx1)
    in_ring_idy1 = conn1.inRing(idy1)

    # Whether each atom is on a ring in both end states.
    on_ring_idx0 = _onRing(idx0, conn0)
    on_ring_idy0 = _onRing(idy0, conn0)
    on_ring_idx1 = _onRing(idx1, conn1)
    on_ring_idy1 = _onRing(idy1, conn1)

    # Both atoms are in a ring in one end state and at least one isn't in the other.
    if (in_ring_idx0 & in_ring_idy0) ^ (in_ring_idx1 & in_ring_idy1):
        return True

    # Both atoms are on a ring in one end state and at least one isn't in the other.
    if ((on_ring_idx0 & on_ring_idy0 & (conn0.connectionType(idx0, idy0) == 4))
        ^ (on_ring_idx1 & on_ring_idy1 & (conn1.connectionType(idx1, idy1) == 4))):
        return True

    # Both atoms are in or on a ring in one state and at least one isn't in the other.
    if (((in_ring_idx0 | on_ring_idx0) & (in_ring_idy0 | on_ring_idy0) & (conn0.connectionType(idx0, idy0) == 3)) ^
        ((in_ring_idx1 | on_ring_idx1) & (in_ring_idy1 | on_ring_idy1) & (conn1.connectionType(idx1, idy1) == 3))):
        iscn0 = set(conn0.connectionsTo(idx0)).intersection(set(conn0.connectionsTo(idy0)))
        if (len(iscn0) != 1):
            return True
        common_idx = iscn0.pop()
        in_ring_bond0 = (conn0.inRing(idx0, common_idx) | conn0.inRing(idy0, common_idx))
        iscn1 = set(conn1.connectionsTo(idx1)).intersection(set(conn1.connectionsTo(idy1)))
        if (len(iscn1) != 1):
            return True
        common_idx = iscn1.pop()
        in_ring_bond1 = (conn1.inRing(idx1, common_idx) | conn1.inRing(idy1, common_idx))
        if (in_ring_bond0 ^ in_ring_bond1):
            return True

    # If we get this far, then a ring wasn't broken.
    return False

def _is_ring_size_changed(conn0, conn1, idx0, idy0, idx1, idy1, max_ring_size=12):
    """Internal function to test whether a perturbation changes the connectivity
       around two atoms such that a ring changes size.
       Parameters
       ----------
       conn0 : Sire.Mol.Connectivity
           The connectivity object for the first end state.
       conn1 : Sire.Mol.Connectivity
           The connectivity object for the second end state.
       idx0 : Sire.Mol.AtomIdx
           The index of the first atom in the first state.
       idy0 : Sire.Mol.AtomIdx
           The index of the second atom in the first state.
       idx1 : Sire.Mol.AtomIdx
           The index of the first atom in the second state.
       idy1 : Sire.Mol.AtomIdx
           The index of the second atom in the second state.
       max_ring_size : int
           The maximum size of what is considered to be a ring.
    """

    # Have a ring changed size? If so, then the minimum path size between
    # two atoms will have changed.

    # Work out the paths connecting the atoms in the two end states.
    paths0 = conn0.findPaths(idx0, idy0, max_ring_size)
    paths1 = conn1.findPaths(idx1, idy1, max_ring_size)

    # Initalise the ring size in each end state.
    ring0 = None
    ring1 = None

    # Determine the minimum path in the lambda = 0 state.
    if len(paths0) > 1:
        path_lengths0 = []
        for path in paths0:
            path_lengths0.append(len(path))
        ring0 = min(path_lengths0)

    if ring0 is None:
        return False

    # Determine the minimum path in the lambda = 1 state.
    if len(paths1) > 1:
        path_lengths1 = []
        for path in paths1:
            path_lengths1.append(len(path))
        ring1 = min(path_lengths1)

    # Return whether the ring has changed size.
    if ring1:
        return ring0 != ring1
    else:
        return False


def _onRing(idx, conn):
    """Internal function to test whether an atom is adjacent to a ring.
       Parameters
       ----------
       idx : Sire.Mol.AtomIdx
           The index of the atom
       conn : Sire.Mol.Connectivity
           The connectivity object.
       Returns
       -------
       is_on_ring : bool
           Whether the atom is adjacent to a ring.
    """

    # Loop over all atoms connected to this atom.
    for x in conn.connectionsTo(idx):
        # The neighbour is in a ring.
        if conn.inRing(x) and (not conn.inRing(x, idx)):
            return True

    # If we get this far, then the atom is not adjacent to a ring.
    return False



def _merge_from_xml(lig1, lig2, mapping, allow_ring_breaking=False, property_map0={}, property_map1={}):
    """Merge this molecule with 'other'.
       Parameters
       ----------
       other : BioSimSpace._SireWrappers.Molecule
           The molecule to merge with.
       mapping : dict
           The mapping between matching atom indices in the two molecules.
       allow_ring_breaking : bool
           Whether to allow the opening/closing of rings during a merge.
       property_map0 : dict
           A dictionary that maps "properties" in this molecule to their
           user defined values. This allows the user to refer to properties
           with their own naming scheme, e.g. { "charge" : "my-charge" }
       property_map1 : dict
           A dictionary that maps "properties" in ther other molecule to
           their user defined values.
       Returns
       -------
       merged : Sire.Mol.Molecule
           The merged molecule.
    """

    # Cannot merge an already merged molecule.
    # if self._is_merged:
    #     raise IncompatibleError("This molecule has already been merged!")
    # if other._is_merged:
    #     raise IncompatibleError("'other' has already been merged!")

    # # Validate input.

    # if type(other) is not Molecule:
    #     raise TypeError("'other' must be of type 'BioSimSpace._SireWrappers.Molecule'")

    if type(property_map0) is not dict:
        raise TypeError("'property_map0' must be of type 'dict'")

    if type(property_map1) is not dict:
        raise TypeError("'property_map1' must be of type 'dict'")

    if type(mapping) is not dict:
        raise TypeError("'mapping' must be of type 'dict'.")
    else:
        # Make sure all key/value pairs are of type AtomIdx.
        for idx0, idx1 in mapping.items():
            if type(idx0) is not _SireMol.AtomIdx or type(idx1) is not _SireMol.AtomIdx:
                raise TypeError("key:value pairs in 'mapping' must be of type 'Sire.Mol.AtomIdx'")

    # Create a copy of this molecule.
    mol = Molecule(lig1._sire_object)

    # Set the two molecule objects.
    molecule0 = lig1._sire_object
    molecule1 = lig2._sire_object
    # molecule0 = lig1.first().molecule()
    # molecule1 = lig2.first().molecule()
    # Get the atom indices from the mapping.
    idx0 = mapping.keys()
    idx1 = mapping.values()

    # Create the reverse mapping: molecule1 --> molecule0
    inv_mapping = {v: k for k, v in mapping.items()}

    # Invert the user property mappings.
    inv_property_map0 = {v: k for k, v in property_map0.items()}
    inv_property_map1 = {v: k for k, v in property_map1.items()}

    # Make sure that the molecules have a "forcefield" property and that
    # the two force fields are compatible.

    # Get the user name for the "forcefield" property.
    # ff0 = inv_property_map0.get("forcefield", "forcefield")
    # ff1 = inv_property_map1.get("forcefield", "forcefield")

    # # Force field information is missing.
    # if not molecule0.hasProperty(ff0):
    #     raise _IncompatibleError("Cannot determine 'forcefield' of 'molecule0'!")
    # if not molecule1.hasProperty(ff0):
    #     raise _IncompatibleError("Cannot determine 'forcefield' of 'molecule1'!")

    # # The force fields are incompatible.
    # if not molecule0.property(ff0).isCompatibleWith(molecule1.property(ff1)):
    #     raise _IncompatibleError("Cannot merge molecules with incompatible force fields!")

    # Create lists to store the atoms that are unique to each molecule,
    # along with their indices.
    atoms0 = []
    atoms1 = []
    atoms0_idx = []
    atoms1_idx = []

    # Loop over each molecule to find the unique atom indices.

    # molecule0
    for atom in molecule0.atoms():
        if atom.index() not in idx0:
            atoms0.append(atom)
            atoms0_idx.append(atom.index())

    # molecule1
    for atom in molecule1.atoms():
        if atom.index() not in idx1:
            atoms1.append(atom)
            atoms1_idx.append(atom.index())

    # Create lists of the actual property names in the molecules.
    props0 = []
    props1 = []

    # molecule0
    for prop in molecule0.propertyKeys():
        if prop in inv_property_map0:
            prop = inv_property_map0[prop]
        props0.append(prop)

    # molecule1
    for prop in molecule1.propertyKeys():
        if prop in inv_property_map1:
            prop = inv_property_map1[prop]
        props1.append(prop)

    # Determine the common properties between the two molecules.
    # These are the properties that can be perturbed.
    shared_props = list(set(props0).intersection(props1))
    del props0
    del props1

    # Create a new molecule to hold the merged molecule.
    molecule = _SireMol.Molecule("Merged_Molecule")

    # Add a single residue called LIG.
    res = molecule.edit().add(_SireMol.ResNum(1))
    res.rename(_SireMol.ResName("LIG"))

    # Create a single cut-group.
    cg = res.molecule().add(_SireMol.CGName("1"))

    # Counter for the number of atoms.
    num = 1

    # First add all of the atoms from molecule0.
    for atom in molecule0.atoms():
        # Add the atom.
        added = cg.add(atom.name())
        added.renumber(_SireMol.AtomNum(num))
        added.reparent(_SireMol.ResIdx(0))
        num += 1

    # Now add all of the atoms from molecule1 that aren't mapped from molecule0.
    for atom in atoms1:
        added = cg.add(atom.name())
        added.renumber(_SireMol.AtomNum(num))
        added.reparent(_SireMol.ResIdx(0))
        inv_mapping[atom.index()] = _SireMol.AtomIdx(num-1)
        num += 1

    # Commit the changes to the molecule.
    molecule = cg.molecule().commit()

    # Make the molecule editable.
    edit_mol = molecule.edit()

    # We now add properties to the merged molecule. The properties are used
    # to represent the molecule at two states along the alchemical pathway:
    #
    # lambda = 0:
    #   Here only molecule0 is active. The "charge" and "LJ" properties
    #   for atoms that are part of molecule1 are set to zero. Other force
    #   field properties, e.g. "bond", "angle", "dihedral", and "improper",
    #   are retained for the atoms in molecule1, although the indices of
    #   the atoms involved in the interactions must be re-mapped to their
    #   positions in the merged molecule. (Also, the interactions may now
    #   be between atoms of different type.) The properties are given the
    #   suffix "0", e.g. "charge0".
    #
    # lambda = 1:
    #   Here only molecule1 is active. We perform the same process as above,
    #   only we modify the properties of the atoms that are unique to
    #   molecule0. The properties are given the suffix "1", e.g. "charge1".
    #
    # Properties that aren't shared between the molecules (and thus can't
    # be merged) are set using their original names.

    ##############################
    # SET PROPERTIES AT LAMBDA = 0
    ##############################

    # Add the atom properties from molecule0.
    for atom in molecule0.atoms():
        # Add an "name0" property.
        edit_mol = edit_mol.atom(atom.index()) \
                           .setProperty("name0", atom.name().value()).molecule()

        # Loop over all atom properties.
        for prop in atom.propertyKeys():
            # Get the actual property name.
            name = inv_property_map0.get(prop, prop)

            # This is a perturbable property. Rename to "property0", e.g. "charge0".
            if name in shared_props:
                name = name + "0"

            # Add the property to the atom in the merged molecule.
            edit_mol = edit_mol.atom(atom.index()).setProperty(name, atom.property(prop)).molecule()

    # Add the atom properties from molecule1.
    for atom in atoms1:
        # Get the atom index in the merged molecule.
        idx = inv_mapping[atom.index()]

        # Add an "name0" property.
        edit_mol = edit_mol.atom(idx).setProperty("name0", atom.name().value()).molecule()

        # Loop over all atom properties.
        for prop in atom.propertyKeys():
            # Get the actual property name.
            name = inv_property_map1.get(prop, prop)

            # Zero the "charge" and "LJ" property for atoms that are unique to molecule1.
            if name == "charge":
                edit_mol = edit_mol.atom(idx).setProperty("charge0", 0*_SireUnits.e_charge).molecule()
            elif name == "LJ":
                edit_mol = edit_mol.atom(idx).setProperty("LJ0", _SireMM.LJParameter()).molecule()
            elif name == "ambertype":
                edit_mol = edit_mol.atom(idx).setProperty("ambertype0", "du").molecule()
            elif name == "element":
                edit_mol = edit_mol.atom(idx).setProperty("element0", _SireMol.Element(0)).molecule()
            else:
                # This is a perturbable property. Rename to "property0", e.g. "charge0".
                if name in shared_props:
                    name = name + "0"

                # Add the property to the atom in the merged molecule.
                edit_mol = edit_mol.atom(idx).setProperty(name, atom.property(prop)).molecule()

    # We now need to merge "bond", "angle", "dihedral", and "improper" parameters.
    # To do so, we extract the properties from molecule0, then add the additional
    # properties from molecule1, making sure to update the atom indices, and bond
    # atoms from molecule1 to the atoms to which they map in molecule0.

    # 1) bonds
    if "bond" in shared_props:
        # Get the user defined property names.
        prop0 = inv_property_map0.get("bond", "bond")
        prop1 = inv_property_map1.get("bond", "bond")

        # Get the "bond" property from the two molecules.
        bonds0 = molecule0.property(prop0)
        bonds1 = molecule1.property(prop1)

        # Get the molInfo object for molecule1.
        info = molecule1.info()

        # Create the new set of bonds.
        bonds = _SireMM.TwoAtomFunctions(edit_mol.info())

        # Add all of the bonds from molecule0.
        for bond in bonds0.potentials():
            bonds.set(bond.atom0(), bond.atom1(), bond.function())

        # Loop over all bonds in molecule1.
        for bond in bonds1.potentials():
            # This bond contains an atom that is unique to molecule1.
            if info.atomIdx(bond.atom0()) in atoms1_idx or \
               info.atomIdx(bond.atom1()) in atoms1_idx:

                # Extract the bond information.
                atom0 = info.atomIdx(bond.atom0())
                atom1 = info.atomIdx(bond.atom1())
                exprn = bond.function()

                # Map the atom indices to their position in the merged molecule.
                atom0 = inv_mapping[atom0]
                atom1 = inv_mapping[atom1]

                # Set the new bond.
                bonds.set(atom0, atom1, exprn)

        # Add the bonds to the merged molecule.
        edit_mol.setProperty("bond0", bonds)

    # 2) angles
    if "angle" in shared_props:
        # Get the user defined property names.
        prop0 = inv_property_map0.get("angle", "angle")
        prop1 = inv_property_map1.get("angle", "angle")

        # Get the "angle" property from the two molecules.
        angles0 = molecule0.property(prop0)
        angles1 = molecule1.property(prop1)

        # Get the molInfo object for molecule1.
        info = molecule1.info()

        # Create the new set of angles.
        angles = _SireMM.ThreeAtomFunctions(edit_mol.info())

        # Add all of the angles from molecule0.
        for angle in angles0.potentials():
            angles.set(angle.atom0(), angle.atom1(), angle.atom2(), angle.function())

        # Loop over all angles in molecule1.
        for angle in angles1.potentials():
            # This angle contains an atom that is unique to molecule1.
            if info.atomIdx(angle.atom0()) in atoms1_idx or \
               info.atomIdx(angle.atom1()) in atoms1_idx or \
               info.atomIdx(angle.atom2()) in atoms1_idx:

                   # Extract the angle information.
                   atom0 = info.atomIdx(angle.atom0())
                   atom1 = info.atomIdx(angle.atom1())
                   atom2 = info.atomIdx(angle.atom2())
                   exprn = angle.function()

                   # Map the atom indices to their position in the merged molecule.
                   atom0 = inv_mapping[atom0]
                   atom1 = inv_mapping[atom1]
                   atom2 = inv_mapping[atom2]

                   # Set the new angle.
                   angles.set(atom0, atom1, atom2, exprn)

        # Add the angles to the merged molecule.
        edit_mol.setProperty("angle0", angles)

    # 3) dihedrals
    if "dihedral" in shared_props:
        # Get the user defined property names.
        prop0 = inv_property_map0.get("dihedral", "dihedral")
        prop1 = inv_property_map1.get("dihedral", "dihedral")

        # Get the "dihedral" property from the two molecules.
        dihedrals0 = molecule0.property(prop0)
        dihedrals1 = molecule1.property(prop1)

        # Get the molInfo object for molecule1.
        info = molecule1.info()

        # Create the new set of dihedrals.
        dihedrals = _SireMM.FourAtomFunctions(edit_mol.info())

        # Add all of the dihedrals from molecule0.
        for dihedral in dihedrals0.potentials():
            dihedrals.set(dihedral.atom0(), dihedral.atom1(),
                dihedral.atom2(), dihedral.atom3(), dihedral.function())

        # Loop over all dihedrals in molecule1.
        for dihedral in dihedrals1.potentials():
            # This dihedral contains an atom that is unique to molecule1.
            if info.atomIdx(dihedral.atom0()) in atoms1_idx or \
               info.atomIdx(dihedral.atom1()) in atoms1_idx or \
               info.atomIdx(dihedral.atom2()) in atoms1_idx or \
               info.atomIdx(dihedral.atom3()) in atoms1_idx:

                   # Extract the dihedral information.
                   atom0 = info.atomIdx(dihedral.atom0())
                   atom1 = info.atomIdx(dihedral.atom1())
                   atom2 = info.atomIdx(dihedral.atom2())
                   atom3 = info.atomIdx(dihedral.atom3())
                   exprn = dihedral.function()

                   # Map the atom indices to their position in the merged molecule.
                   atom0 = inv_mapping[atom0]
                   atom1 = inv_mapping[atom1]
                   atom2 = inv_mapping[atom2]
                   atom3 = inv_mapping[atom3]

                   # Set the new dihedral.
                   dihedrals.set(atom0, atom1, atom2, atom3, exprn)

        # Add the dihedrals to the merged molecule.
        edit_mol.setProperty("dihedral0", dihedrals)

    # 4) impropers
    if "improper" in shared_props:
        # Get the user defined property names.
        prop0 = inv_property_map0.get("improper", "improper")
        prop1 = inv_property_map1.get("improper", "improper")

        # Get the "improper" property from the two molecules.
        impropers0 = molecule0.property(prop0)
        impropers1 = molecule1.property(prop1)

        # Get the molInfo object for molecule1.
        info = molecule1.info()

        # Create the new set of impropers.
        impropers = _SireMM.FourAtomFunctions(edit_mol.info())

        # Add all of the impropers from molecule0.
        for improper in impropers0.potentials():
            impropers.set(improper.atom0(), improper.atom1(),
                improper.atom2(), improper.atom3(), improper.function())

        # Loop over all impropers in molecule1.
        for improper in impropers1.potentials():
            # This improper contains an atom that is unique to molecule1.
            if info.atomIdx(improper.atom0()) in atoms1_idx or \
               info.atomIdx(improper.atom1()) in atoms1_idx or \
               info.atomIdx(improper.atom2()) in atoms1_idx or \
               info.atomIdx(improper.atom3()) in atoms1_idx:

                   # Extract the improper information.
                   atom0 = info.atomIdx(improper.atom0())
                   atom1 = info.atomIdx(improper.atom1())
                   atom2 = info.atomIdx(improper.atom2())
                   atom3 = info.atomIdx(improper.atom3())
                   exprn = improper.function()

                   # Map the atom indices to their position in the merged molecule.
                   atom0 = inv_mapping[atom0]
                   atom1 = inv_mapping[atom1]
                   atom2 = inv_mapping[atom2]
                   atom3 = inv_mapping[atom3]

                   # Set the new improper.
                   impropers.set(atom0, atom1, atom2, atom3, exprn)

        # Add the impropers to the merged molecule.
        edit_mol.setProperty("improper0", impropers)

    ##############################
    # SET PROPERTIES AT LAMBDA = 1
    ##############################

    # Add the atom properties from molecule1.
    for atom in molecule1.atoms():
        # Get the atom index in the merged molecule.
        idx = inv_mapping[atom.index()]

        # Add an "name1" property.
        edit_mol = edit_mol.atom(idx).setProperty("name1", atom.name().value()).molecule()

        # Loop over all atom properties.
        for prop in atom.propertyKeys():
            # Get the actual property name.
            name = inv_property_map1.get(prop, prop)

            # This is a perturbable property. Rename to "property1", e.g. "charge1".
            if name in shared_props:
                name = name + "1"

            # Add the property to the atom in the merged molecule.
            edit_mol = edit_mol.atom(idx).setProperty(name, atom.property(prop)).molecule()

    # Add the properties from atoms unique to molecule0.
    for atom in atoms0:
        # Add an "name1" property.
        edit_mol = edit_mol.atom(atom.index()) \
                           .setProperty("name1", atom.name().value()).molecule()

        # Loop over all atom properties.
        for prop in atom.propertyKeys():
            # Get the actual property name.
            name = inv_property_map0.get(prop, prop)

            # Zero the "charge" and "LJ" property for atoms that are unique to molecule0.
            if name == "charge":
                edit_mol = edit_mol.atom(atom.index()).setProperty("charge1", 0*_SireUnits.e_charge).molecule()
            elif name == "LJ":
                edit_mol = edit_mol.atom(atom.index()).setProperty("LJ1", _SireMM.LJParameter()).molecule()
            elif name == "ambertype":
                edit_mol = edit_mol.atom(atom.index()).setProperty("ambertype1", "du").molecule()
            elif name == "element":
                edit_mol = edit_mol.atom(atom.index()).setProperty("element1", _SireMol.Element(0)).molecule()
            else:
                # This is a perturbable property. Rename to "property1", e.g. "charge1".
                if name in shared_props:
                    name = name + "1"

                # Add the property to the atom in the merged molecule.
                edit_mol = edit_mol.atom(atom.index()).setProperty(name, atom.property(prop)).molecule()

    # We now need to merge "bond", "angle", "dihedral", and "improper" parameters.
    # To do so, we extract the properties from molecule1, then add the additional
    # properties from molecule0, making sure to update the atom indices, and bond
    # atoms from molecule0 to the atoms to which they map in molecule1.

    # 1) bonds
    if "bond" in shared_props:
        # Get the info objects for the two molecules.
        info0 = molecule0.info()
        info1 = molecule1.info()

        # Get the user defined property names.
        prop0 = inv_property_map0.get("bond", "bond")
        prop1 = inv_property_map1.get("bond", "bond")

        # Get the "bond" property from the two molecules.
        bonds0 = molecule0.property(prop0)
        bonds1 = molecule1.property(prop1)

        # Create the new set of bonds.
        bonds = _SireMM.TwoAtomFunctions(edit_mol.info())

        # Add all of the bonds from molecule1.
        for bond in bonds1.potentials():
            # Extract the bond information.
            atom0 = info1.atomIdx(bond.atom0())
            atom1 = info1.atomIdx(bond.atom1())
            exprn = bond.function()

            # Map the atom indices to their position in the merged molecule.
            atom0 = inv_mapping[atom0]
            atom1 = inv_mapping[atom1]

            # Set the new bond.
            bonds.set(atom0, atom1, exprn)

        # Loop over all bonds in molecule0
        for bond in bonds0.potentials():
            # This bond contains an atom that is unique to molecule0.
            if info0.atomIdx(bond.atom0()) in atoms0_idx or \
               info0.atomIdx(bond.atom1()) in atoms0_idx:

               # Extract the bond information.
               atom0 = info0.atomIdx(bond.atom0())
               atom1 = info0.atomIdx(bond.atom1())
               exprn = bond.function()

               # Set the new bond.
               bonds.set(atom0, atom1, exprn)

        # Add the bonds to the merged molecule.
        edit_mol.setProperty("bond1", bonds)

    # 2) angles
    if "angle" in shared_props:
        # Get the info objects for the two molecules.
        info0 = molecule0.info()
        info1 = molecule1.info()

        # Get the user defined property names.
        prop0 = inv_property_map0.get("angle", "angle")
        prop1 = inv_property_map1.get("angle", "angle")

        # Get the "angle" property from the two molecules.
        angles0 = molecule0.property(prop0)
        angles1 = molecule1.property(prop1)

        # Create the new set of angles.
        angles = _SireMM.ThreeAtomFunctions(edit_mol.info())

        # Add all of the angles from molecule1.
        for angle in angles1.potentials():
            # Extract the angle information.
            atom0 = info1.atomIdx(angle.atom0())
            atom1 = info1.atomIdx(angle.atom1())
            atom2 = info1.atomIdx(angle.atom2())
            exprn = angle.function()

            # Map the atom indices to their position in the merged molecule.
            atom0 = inv_mapping[atom0]
            atom1 = inv_mapping[atom1]
            atom2 = inv_mapping[atom2]

            # Set the new angle.
            angles.set(atom0, atom1, atom2, exprn)

        # Loop over all angles in molecule0.
        for angle in angles0.potentials():
            # This angle contains an atom that is unique to molecule0.
            if info0.atomIdx(angle.atom0()) in atoms0_idx or \
               info0.atomIdx(angle.atom1()) in atoms0_idx or \
               info0.atomIdx(angle.atom2()) in atoms0_idx:

                   # Extract the angle information.
                   atom0 = info0.atomIdx(angle.atom0())
                   atom1 = info0.atomIdx(angle.atom1())
                   atom2 = info0.atomIdx(angle.atom2())
                   exprn = angle.function()

                   # Set the new angle.
                   angles.set(atom0, atom1, atom2, exprn)

        # Add the angles to the merged molecule.
        edit_mol.setProperty("angle1", angles)

    # 3) dihedrals
    if "dihedral" in shared_props:
        # Get the info objects for the two molecules.
        info0 = molecule0.info()
        info1 = molecule1.info()

        # Get the user defined property names.
        prop0 = inv_property_map0.get("dihedral", "dihedral")
        prop1 = inv_property_map1.get("dihedral", "dihedral")

        # Get the "dihedral" property from the two molecules.
        dihedrals0 = molecule0.property(prop0)
        dihedrals1 = molecule1.property(prop1)

        # Create the new set of dihedrals.
        dihedrals = _SireMM.FourAtomFunctions(edit_mol.info())

        # Add all of the dihedrals from molecule1.
        for dihedral in dihedrals1.potentials():
            # Extract the dihedral information.
            atom0 = info1.atomIdx(dihedral.atom0())
            atom1 = info1.atomIdx(dihedral.atom1())
            atom2 = info1.atomIdx(dihedral.atom2())
            atom3 = info1.atomIdx(dihedral.atom3())
            exprn = dihedral.function()

            # Map the atom indices to their position in the merged molecule.
            atom0 = inv_mapping[atom0]
            atom1 = inv_mapping[atom1]
            atom2 = inv_mapping[atom2]
            atom3 = inv_mapping[atom3]

            # Set the new dihedral.
            dihedrals.set(atom0, atom1, atom2, atom3, exprn)

        # Loop over all dihedrals in molecule0.
        for dihedral in dihedrals0.potentials():
            # This dihedral contains an atom that is unique to molecule0.
            if info0.atomIdx(dihedral.atom0()) in atoms0_idx or \
               info0.atomIdx(dihedral.atom1()) in atoms0_idx or \
               info0.atomIdx(dihedral.atom2()) in atoms0_idx or \
               info0.atomIdx(dihedral.atom3()) in atoms0_idx:

                   # Extract the dihedral information.
                   atom0 = info0.atomIdx(dihedral.atom0())
                   atom1 = info0.atomIdx(dihedral.atom1())
                   atom2 = info0.atomIdx(dihedral.atom2())
                   atom3 = info0.atomIdx(dihedral.atom3())
                   exprn = dihedral.function()

                   # Set the new dihedral.
                   dihedrals.set(atom0, atom1, atom2, atom3, exprn)

        # Add the dihedrals to the merged molecule.
        edit_mol.setProperty("dihedral1", dihedrals)

    # 4) impropers
    if "improper" in shared_props:
        # Get the info objects for the two molecules.
        info0 = molecule0.info()
        info1 = molecule1.info()

        # Get the user defined property names.
        prop0 = inv_property_map0.get("improper", "improper")
        prop1 = inv_property_map1.get("improper", "improper")

        # Get the "improper" property from the two molecules.
        impropers0 = molecule0.property(prop0)
        impropers1 = molecule1.property(prop1)

        # Create the new set of impropers.
        impropers = _SireMM.FourAtomFunctions(edit_mol.info())

        # Add all of the impropers from molecule1.
        for improper in impropers1.potentials():
            # Extract the improper information.
            atom0 = info1.atomIdx(improper.atom0())
            atom1 = info1.atomIdx(improper.atom1())
            atom2 = info1.atomIdx(improper.atom2())
            atom3 = info1.atomIdx(improper.atom3())
            exprn = improper.function()

            # Map the atom indices to their position in the merged molecule.
            atom0 = inv_mapping[atom0]
            atom1 = inv_mapping[atom1]
            atom2 = inv_mapping[atom2]
            atom3 = inv_mapping[atom3]

            # Set the new improper.
            impropers.set(atom0, atom1, atom2, atom3, exprn)

        # Loop over all impropers in molecule0.
        for improper in impropers0.potentials():
            # This improper contains an atom that is unique to molecule0.
            if info0.atomIdx(improper.atom0()) in atoms0_idx or \
               info0.atomIdx(improper.atom1()) in atoms0_idx or \
               info0.atomIdx(improper.atom2()) in atoms0_idx or \
               info0.atomIdx(improper.atom3()) in atoms0_idx:

                   # Extract the improper information.
                   atom0 = info0.atomIdx(improper.atom0())
                   atom1 = info0.atomIdx(improper.atom1())
                   atom2 = info0.atomIdx(improper.atom2())
                   atom3 = info0.atomIdx(improper.atom3())
                   exprn = improper.function()

                   # Set the new improper.
                   impropers.set(atom0, atom1, atom2, atom3, exprn)

        # Add the impropers to the merged molecule.
        edit_mol.setProperty("improper1", impropers)

    # The number of potentials should be consistent for the "bond0"
    # and "bond1" properties.
    if edit_mol.property("bond0").nFunctions() != edit_mol.property("bond1").nFunctions():
        raise _IncompatibleError("Inconsistent number of bonds in merged molecule!")

    # Create the connectivity object
    conn = _SireMol.Connectivity(edit_mol.info()).edit()

    # Connect the bonded atoms. Connectivity is the same at lambda = 0
    # and lambda = 1.
    for bond in edit_mol.property("bond0").potentials():
        conn.connect(bond.atom0(), bond.atom1())
    conn = conn.commit()

    # Get the connectivity of the two molecules.
    c0 = molecule0.property("connectivity")
    c1 = molecule1.property("connectivity")

    # Check that the merge hasn't modified the connectivity.

    # molecule0
    for x in range(0, molecule0.nAtoms()):
        # Convert to an AtomIdx.
        idx = _SireMol.AtomIdx(x)

        for y in range(x+1, molecule0.nAtoms()):
            # Convert to an AtomIdx.
            idy = _SireMol.AtomIdx(y)

            # The connectivity has changed.
            if c0.connectionType(idx, idy) != conn.connectionType(idx, idy):
                # Ring opening/closing is allowed.
                if allow_ring_breaking:
                    if not _is_ring_broken(c0, conn, idx, idy, idx, idy):
                        raise _IncompatibleError("The merge has changed the molecular connectivity! "
                                                 "Check your atom mapping.")
                else:
                    raise _IncompatibleError("The merge has changed the molecular connectivity! "
                                             "If you want to open/close a ring, then set the "
                                             "'allow_ring_breaking' option to 'True'.")

            # Check that a ring hasn't been opened/closed.
            else:
                if not allow_ring_breaking:
                    # We require that both atoms are in a ring before and aren't after, or vice-versa.
                    if _is_ring_broken(c0, conn, idx, idy, idx, idy):
                        raise _IncompatibleError("The merge has changed opened/closed a ring! "
                                                "If you want to allow this perturbation, then set the "
                                                "'allow_ring_breaking' option to 'True'.")

    # molecule1
    for x in range(0, molecule1.nAtoms()):
        # Convert to an AtomIdx.
        idx = _SireMol.AtomIdx(x)

        # Map the index to its position in the merged molecule.
        idx_map = inv_mapping[idx]

        for y in range(x+1, molecule1.nAtoms()):
            # Convert to an AtomIdx.
            idy = _SireMol.AtomIdx(y)

            # Map the index to its position in the merged molecule.
            idy_map = inv_mapping[idy]

            # The connectivity has changed.
            if c1.connectionType(idx, idy) != conn.connectionType(idx_map, idy_map):
                # Ring opening/closing is forbidden.
                if allow_ring_breaking:
                    # We require that both atoms are in a ring before and aren't after, or vice-versa.
                    if not _is_ring_broken(c1, conn, idx, idy, idx_map, idy_map):
                        raise _IncompatibleError("The merge has changed the molecular connectivity! "
                                                 "Check your atom mapping.")
                else:
                    raise _IncompatibleError("The merge has changed the molecular connectivity! "
                                             "If you want to open/close a ring, then set the "
                                             "'allow_ring_breaking' option to 'True'.")

            # Check that a ring hasn't been opened/closed.
            else:
                if not allow_ring_breaking:
                    # We require that both atoms are in a ring before and aren't after, or vice-versa.
                    if _is_ring_broken(c1, conn, idx, idy, idx_map, idy_map):
                        raise _IncompatibleError("The merge has changed opened/closed a ring! "
                                                "If you want to allow this perturbation, then set the "
                                                "'allow_ring_breaking' option to 'True'.")

    # Set the "connectivity" property.
    edit_mol.setProperty("connectivity", conn)

    # Create the CLJNBPairs matrices.
    #ff = molecule0.property(ff0)

    clj_nb_pairs0 = _SireMM.CLJNBPairs(edit_mol.info(),
        _SireMM.CLJScaleFactor(0, 0))

    # Loop over all atoms unique to molecule0.
    for idx0 in atoms0_idx:
        # Loop over all atoms unique to molecule1.
        for idx1 in atoms1_idx:
            # Map the index to its position in the merged molecule.
            idx1 = inv_mapping[idx1]

            # Work out the connection type between the atoms.
            conn_type = conn.connectionType(idx0, idx1)

            # The atoms aren't bonded.
            if conn_type == 0:
                clj_scale_factor = _SireMM.CLJScaleFactor(1, 1)
                clj_nb_pairs0.set(idx0, idx1, clj_scale_factor)

            # The atoms are part of a dihedral.
            elif conn_type == 4:
                clj_scale_factor = _SireMM.CLJScaleFactor(0.5 ,
                                                          0.5)
                clj_nb_pairs0.set(idx0, idx1, clj_scale_factor)

    # Copy the intrascale matrix.
    clj_nb_pairs1 = clj_nb_pairs0.__deepcopy__()

    # Get the user defined "intrascale" property names.
    prop0 = inv_property_map0.get("intrascale", "intrascale")
    prop1 = inv_property_map1.get("intrascale", "intrascale")

    # Get the "intrascale" property from the two molecules.
    intrascale0 = molecule0.property(prop0)
    intrascale1 = molecule1.property(prop1)

    # Copy the intrascale from molecule1 into clj_nb_pairs0.

    # Perform a triangular loop over atoms from molecule1.
    for x in range(0, molecule1.nAtoms()):
        # Convert to an AtomIdx.
        idx = _SireMol.AtomIdx(x)

        # Map the index to its position in the merged molecule.
        idx = inv_mapping[idx]

        for y in range(x+1, molecule1.nAtoms()):
            # Convert to an AtomIdx.
            idy = _SireMol.AtomIdx(y)

            # Map the index to its position in the merged molecule.
            idy = inv_mapping[idy]

            # Get the intrascale value.
            intra = intrascale1.get(_SireMol.AtomIdx(x), _SireMol.AtomIdx(y))

            # Only set if there is a non-zero value.
            # Set using the re-mapped atom indices.
            if not intra.coulomb() == 0:
                clj_nb_pairs0.set(idx, idy, intra)

    # Now copy in all intrascale values from molecule0 into both
    # clj_nb_pairs matrices.

    # Perform a triangular loop over atoms from molecule0.
    for x in range(0, molecule0.nAtoms()):
        for y in range(x+1, molecule0.nAtoms()):
            # Get the intrascale value.
            intra = intrascale0.get(_SireMol.AtomIdx(x), _SireMol.AtomIdx(y))

            # Set the value in the new matrix, overwriting existing value.
            clj_nb_pairs0.set(_SireMol.AtomIdx(x), _SireMol.AtomIdx(y), intra)

            # Only set if there is a non-zero value.
            if not intra.coulomb() == 0:
                clj_nb_pairs1.set(_SireMol.AtomIdx(x), _SireMol.AtomIdx(y), intra)

    # Finally, copy the intrascale from molecule1 into clj_nb_pairs1.

    # Perform a triangular loop over atoms from molecule1.
    for x in range(0, molecule1.nAtoms()):
        # Convert to an AtomIdx.
        idx = _SireMol.AtomIdx(x)

        # Map the index to its position in the merged molecule.
        idx = inv_mapping[idx]

        for y in range(x+1, molecule1.nAtoms()):
            # Convert to an AtomIdx.
            idy = _SireMol.AtomIdx(y)

            # Map the index to its position in the merged molecule.
            idy = inv_mapping[idy]

            # Get the intrascale value.
            intra = intrascale1.get(_SireMol.AtomIdx(x), _SireMol.AtomIdx(y))

            # Set the value in the new matrix, overwriting existing value.
            clj_nb_pairs1.set(idx, idy, intra)

    # Store the two molecular components.
    edit_mol.setProperty("molecule0", molecule0)
    edit_mol.setProperty("molecule1", molecule1)

    # Set the "intrascale" properties.
    edit_mol.setProperty("intrascale0", clj_nb_pairs0)
    edit_mol.setProperty("intrascale1", clj_nb_pairs1)

    # Set the "forcefield" properties.
    # edit_mol.setProperty("forcefield0", molecule0.property(ff0))
    # edit_mol.setProperty("forcefield1", molecule1.property(ff1))

    # Flag that this molecule is perturbable.
    edit_mol.setProperty("is_perturbable", _SireBase.wrap(True))

    # Update the Sire molecule object of the new molecule.
    mol._sire_object = edit_mol.commit()

    # Flag that the molecule has been merged.
    mol._is_merged = True

    # Store the components of the merged molecule.
    mol._molecule0 = Molecule(molecule0)
    mol._molecule1 = Molecule(molecule1)

    # Return the new molecule.
    return mol


merged = _merge_from_xml(molecule0, molecule1, sire_mapping, allow_ring_breaking=False, property_map0={}, property_map1={})

bss_ligand1 = bss_mol1.getMolecules()[0]       
bss_mol1.removeMolecules(bss_ligand1)
bss_merged = BSS._SireWrappers.Molecule(merged)
bss_mol1.addMolecules(bss_merged)


def writeLog(ligA, ligB, mapping):
    """ Human readable report on atoms used for the mapping."""
    atoms_in_A = list(mapping.keys())
    stream = open('somd.mapping','w')
    atAdone = []
    atBdone= []
    for atAidx in atoms_in_A:
        atA = ligA._sire_object.select(AtomIdx(atAidx))
        atB = ligB._sire_object.select(AtomIdx(mapping[atAidx]))
        stream.write("%s %s --> %s %s\n" % (atA.index(), atA.name(),atB.index(), atB.name()))
        atAdone.append(atA)
        atBdone.append(atB)
    for atom in ligA._sire_object.atoms():
        if atom in atAdone:
            continue
        stream.write("%s %s --> dummy\n" % (atom.index(), atom.name()))
    for atom in ligB._sire_object.atoms():
        if atom in atBdone:
            continue
        stream.write("dummy --> %s %s\n" % (atom.index(), atom.name()))
    stream.close()

writeLog(molecule0, molecule1, mappings)
BSS.IO.saveMolecules("merged_at_lam0_from_xml.pdb", bss_merged, "PDB", { "coordinates" : "coordinates0" , "element": "element0" })

