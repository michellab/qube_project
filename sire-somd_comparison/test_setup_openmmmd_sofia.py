
from Sire.IO import *
from Sire.Move import *
from Sire.Mol import *
from Sire.MM import *
from Sire.System import *
from Sire.Units import *
from Sire.Maths import *
from Sire.CAS import *
from Sire.FF import *
from Sire.Vol import *
from Sire.Qt import *
from Sire.ID import *
from Sire.Config import *
from Sire.Analysis import *
from Sire.Tools.DCDFile import *
from Sire.Tools import Parameter, resolveParameters
import Sire.Stream
import time
import os.path
import numpy as np
import zipfile
import os
import re
import sys

from Sire.Base import *

# Make sure that the OPENMM_PLUGIN_DIR enviroment variable is set correctly.
#os.environ["OPENMM_PLUGIN_DIR"] = getLibDir() + "/plugins"



from nose.tools import assert_almost_equal
plugins = os.path.join('lib','plugins')
try:
    os.environ["OPENMM_PLUGIN_DIR"]
except KeyError:
    os.environ["OPENMM_PLUGIN_DIR"] = os.path.join(Sire.Base.getInstallDir(),plugins)



def readXmlParameters(pdbfile, xmlfile):
# 1) Read a pdb file describing the system to simulate

    p = PDB2(pdbfile)
    s = p.toSystem()
    molecules = s.molecules()
    #print (molecules)
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
    #print (dicts_ang)

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
        #print(opls)

        name=[]
        for i in range (0, int(len(dicts_atom)/2)): 
            nm={} 
            nm = dicts_atom[i]['name'] 
            name.append(nm) 
        #print(name)

        two=[] 
        #print(len(name)) 
        for i in range(0, len(name)): 
            t=(opls[i],name[i]) 
            two.append(t) 
        #print(two)

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

        for atom in atoms: 
            editatom = editmol.atom(atom.index())

            i = int(str(atom.number()).split('(')[1].replace(")" , " "))
            #print(i)
            editatom.setProperty("charge", float(atom_sorted[i-1]['charge']) * mod_electron)
            editatom.setProperty("mass", float(type_sorted[i-1]['mass']) * g_per_mol) 
            editatom.setProperty("LJ", LJParameter( float(atom_sorted[i-1]['sigma'])*10 * angstrom , float(atom_sorted[i-1]['epsilon'])/4.184 * kcal_per_mol))
            editatom.setProperty("ambertype", dicts_atom[i-1]['type'])
           
            editmol = editatom.molecule()
            #print(editmol)
            #print(editmol)
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
         

        # Now we create a connectivity see setConnectivity in SireIO/amber.cpp l2144
        # XML data tells us how atoms are bonded in the molecule (Bond 'from' and 'to')
        
        if natoms > 1:
            print("Set up connectivity")

            con = []
            for i in range(0,int(nbond/2)):
               # i = int(str(atom.number()).split('(')[1].replace(")" , " "))
                if natoms > 1:   
                    connect_prop= {}
                    connect_prop = dicts_bond[i]['from'], dicts_bond[i]['to']
                con.append(connect_prop)
            #print(c)

            conn = Connectivity(editmol.info()).edit()

            for j in range(0,len(con)):
                conn.connect(atoms[int(con[j][0]) ].index(), atoms[int(con[j][1]) ].index()) 
                   
            #for atom in atoms:
            editmol.setProperty("connectivity", conn.commit()).commit()
            #       print(editmol.setProperty("connectivity", conn.commit()))
            mol = editmol.setProperty("connectivity", conn.commit()).commit()
            system.update(mol)

             # Now we add bond parameters to the Sire molecule. We also update amberparameters see SireIO/amber.cpp l2154

            internalff = InternalFF()
                    
            bondfuncs = TwoAtomFunctions(mol)
            r = internalff.symbols().bond().r()

            for j in range(0,len(con)):
                bondfuncs.set(atoms[int(con[j][0]) ].index(), atoms[int(con[j][1]) ].index(), float(dicts_bond[j+len(con)]['k'])/836.8* (float(dicts_bond[j+len(con)]['length'])*10 - r) **2  )
                bond_id = BondID(atoms[int(con[j][0])].index(), atoms[int(con[j][1])].index())
                #print(bond_id)
                mol_params.add(bond_id, float(dicts_bond[j+len(con)]['k'])/(2*100*4.184), float(dicts_bond[j+len(con)]['length'])*10 ) 
                    # mol_params.add(bond_id, float(dicts_bond[i]['k']), float(dicts_bond[i]['length']) ) 
            
                editmol.setProperty("bonds", bondfuncs).commit()
                #mol = editmol.setProperty("bonds", bondfuncs).commit()
                molecule = editmol.commit()
                #system.update(mol)

            mol_params.getAllBonds() 

            editmol.setProperty("amberparameters", mol_params).commit() # Weird, should work - investigate ? 
            molecule = editmol.commit()
            #newmolecules.add(molecule)

        # Now we add angle parameters to the Sire molecule. We also update amberparameters see SireIO/amber.cpp L2172
        if natoms > 2:
            print("Set up angles")

            anglefuncs = ThreeAtomFunctions(mol)

            at1 = []
            for i in range(0, nAngles):
                a1 = {}
                to_str1 = str(re.findall(r"\d+",str(dicts_angle[i]['class1'])))
                a1 = int(to_str1.replace("[","").replace("]","").replace("'","") )-800
                at1.append(a1)
            #print (at1)

            at2 = []
            for i in range(0, nAngles):
                a2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_angle[i]['class2'])))
                a2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                at2.append(a2)
            #print (at2)

            at3 = []
            for i in range(0, nAngles):
                a3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_angle[i]['class3'])))
                a3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800
                at3.append(a3)
            #print (at3)

            theta = internalff.symbols().angle().theta()
            for j in range(0,nAngles):
                anglefuncs.set( atoms[at1[j]].index(), atoms[at2[j]].index(), atoms[at3[j]].index(), float(dicts_angle[j]['k'])/(2*4.184) * ( (float(dicts_angle[j]['angle']) - theta )**2 ))
                angle_id = AngleID( atoms[int(at1[j])].index(), atoms[int(at2[j])].index(), atoms[int(at3[j])].index())
                #print(angle_id)
                mol_params.add(angle_id, float(dicts_angle[j]['k'])/(2*4.184), float(dicts_angle[j]['angle']) ) 
            #print(mol_params.getAllAngles() )

        # Now we add dihedral parameters to the Sire molecule. We also update amberparameters see SireIO/amber.cpp L2190

        if natoms > 3:
            print("Set up dihedrals")
            di1 = []
            for i in range(0, nProper):
                d1 = {}
                to_str1 = str(re.findall(r"\d+",str(dicts_proper[i]['class1'])))
                d1 = int(to_str1.replace("[","").replace("]","").replace("'","") )-800
                di1.append(d1)
            #print (di1)

            di2 = []
            for i in range(0, nProper):
                d2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_proper[i]['class2'])))
                d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                di2.append(d2)
            #print (di2)

            di3 = []
            for i in range(0, nProper):
                d3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_proper[i]['class3'])))
                d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800
                di3.append(d3)
            #print (di3)

            di4 = []
            for i in range(0, nProper):
                d4 = {}
                to_str4 = str(re.findall(r"\d+",str(dicts_proper[i]['class4'])))
                d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )-800
                di4.append(d4)
            #print (di4)


            dihedralfuncs = FourAtomFunctions(mol)

            phi = internalff.symbols().dihedral().phi()
            for i in range(0,nProper):  
                dihedral_id = DihedralID( atoms[int(di1[i])].index(), atoms[int(di2[i])].index(), atoms[int(di3[i])].index(), atoms[int(di4[i])].index())
                #print(dihedral_id) 
                dih1= float(dicts_proper[i]['k1'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity1'])* phi - float(dicts_proper[i]['phase1'])))
                dih2= float(dicts_proper[i]['k2'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity2'])* phi - float(dicts_proper[i]['phase2'])))
                dih3= float(dicts_proper[i]['k3'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity3'])* phi - float(dicts_proper[i]['phase3'])))
                dih4= float(dicts_proper[i]['k4'])/4.184*(1+Cos(int(dicts_proper[i]['periodicity4'])* phi - float(dicts_proper[i]['phase4'])))
                dih_fun = dih1 + dih2 +dih3 +dih4
                dihedralfuncs.set(dihedral_id, dih_fun)
                #print(dihedralfuncs.potentials())
                for t in range(1,5):
                    mol_params.add(dihedral_id, float(dicts_proper[i]['k%s'%t])/4.184, int(dicts_proper[i]['periodicity%s'%t]), float(dicts_proper[i]['phase%s'%t]) ) 
        #print(mol_params.getAllDihedrals() )

            

            print("Set up impropers")

            di_im1 = []
            for i in range(0, nImproper):
                d1 = {}
                to_str1 = str(re.findall(r"\d+",str(dicts_improper[i]['class1'])))
                d1 = int(to_str1.replace("[","").replace("]","").replace("'","") )-800
                di_im1.append(d1)


            di_im2 = []
            for i in range(0, nImproper):
                d2 = {}
                to_str2 = str(re.findall(r"\d+",str(dicts_improper[i]['class2'])))
                d2 = int(to_str2.replace("[","").replace("]","").replace("'","") )-800
                di_im2.append(d2)

            di_im3 = []
            for i in range(0, nImproper):
                d3 = {}
                to_str3 = str(re.findall(r"\d+",str(dicts_improper[i]['class3'])))
                d3 = int(to_str3.replace("[","").replace("]","").replace("'","") )-800
                di_im3.append(d3)

            di_im4 = []
            for i in range(0, nImproper):
                d4 = {}
                to_str4 = str(re.findall(r"\d+",str(dicts_improper[i]['class4'])))
                d4 = int(to_str4.replace("[","").replace("]","").replace("'","") )-800
                di_im4.append(d4)


            improperfuncs = FourAtomFunctions(mol)

            phi_im = internalff.symbols().improper().phi()

            
            for i in range(0,nImproper):  
                improper_id = ImproperID( atoms[int(di_im1[i])].index(), atoms[int(di_im2[i])].index(), atoms[int(di_im3[i])].index(), atoms[int(di_im4[i])].index())
                #print(improper_id) 
                imp1= float(dicts_improper[i]['k1'])/4.184*(1+Cos(int(dicts_improper[i]['periodicity1'])* phi_im - float(dicts_improper[i]['phase1'])))
                imp2= float(dicts_improper[i]['k2'])/4.184*(1+Cos(int(dicts_improper[i]['periodicity2'])* phi_im - float(dicts_improper[i]['phase2'])))
                imp3= float(dicts_improper[i]['k3'])/4.184*(1+Cos(int(dicts_improper[i]['periodicity3'])* phi_im - float(dicts_improper[i]['phase3'])))
                imp4= float(dicts_improper[i]['k4'])/4.184*(1+Cos(int(dicts_improper[i]['periodicity4'])* phi_im - float(dicts_improper[i]['phase4'])))
                imp_fun = imp1 + imp2 +imp3 +imp4
                improperfuncs.set(improper_id, imp_fun)
                print(improperfuncs.potentials())

                for t in range(1,5):
                    mol_params.add(improper_id, float(dicts_improper[i]['k%s'%t])/4.184, int(dicts_improper[i]['periodicity%s'%t]), float(dicts_improper[i]['phase%s'%t]) ) 
        #print(mol_params.getAllDihedrals() )

            mol = editmol.setProperty("bond", bondfuncs).commit()
            mol = editmol.setProperty("angle" , anglefuncs).commit()
            mol = editmol.setProperty("dihedral" , dihedralfuncs).commit()
            mol = editmol.setProperty("improper" , improperfuncs).commit()
            system.update(mol)

        # Now we work out non bonded pairs see SireIO/amber.cpp L2213


            print("Set up nbpairs")

            ## Define the bonded pairs in a list that is called are12
            are12 = []
            for i in range(0, natoms): 
                for j in range (0, natoms): 
                    if conn.areBonded(atoms[i].index(), atoms[j].index()) == True:
                        ij = {}
                        ij= (i,j)
                        are12.append(ij)
            are12_bckup = are12[:]

            #Find the 1-2 pairs that appear twice and erase one of them eg. (2,3) and (3,2)
            for i in range (0, len(are12)): 
                for j in range (0, len(are12)): 
                    if are12[i][0] == are12[j][1] and are12[i][1] == are12[j][0]: 
                        are12[j] = (0,0)

            for item in are12[:]: 
                if item == (0,0):
                    are12.remove(item)

            ## Define the 1-3 pairs in a list that is called are13
            are13 = []
            for i in range(0, natoms): 
                for j in range (0, natoms): 
                    if conn.areAngled(atoms[i].index(), atoms[j].index()) == True:
                        ij = {}
                        ij= (i,j)
                        are13.append(ij)
            are13_bckup = are13[:]

            for i in range (0, len(are13)): 
                for j in range (0, len(are13)): 
                    if are13[i][0] == are13[j][1] and are13[i][1] == are13[j][0]: 
                        are13[j] = (0,0)

            for item in are13[:]: 
                if item == (0,0):
                    are13.remove(item)

            ## Define the 1-4 pairs in a list that is called are14
            are14 = []
            for i in range(0, natoms): 
                for j in range (0, natoms): 
                    if conn.areDihedraled(atoms[i].index(), atoms[j].index()) == True:
                        ij = {}
                        ij= (i,j)
                        are14.append(ij)
            are14_bckup = are14[:]

            for i in range (0, len(are14)): 
                for j in range (0, len(are14)): 
                    if are14[i][0] == are14[j][1] and are14[i][1] == are14[j][0]: 
                        are14[j] = (0,0)

            for item in are14[:]: 
                if item == (0,0):
                    are14.remove(item)


            bonded_pairs_list = are12_bckup + are13_bckup + are14_bckup    
            nb_pair_list =[]

            for i in range(0, natoms): 
                for j in range (0, natoms):
                    if i != j and (i,j) not in bonded_pairs_list:
                        nb_pair_list.append((i,j))


            nbpairs = CLJNBPairs(editmol.info(), CLJScaleFactor(0,0))

            for i in range(0, len(are12)):
                scale_factor1 = 0
                scale_factor2 = 0
                nbpairs.set(atoms.index( int(are12[i][0])), atoms.index(int(are12[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                # print(atoms.index( int(are12[i][0])), atoms.index(int(are12[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                # print (nbpairs)
                # print("~~~~~~~~~~~~~~~~~~`")

            for i in range(0, len(are13)):
                scale_factor1 = 0
                scale_factor2 = 0
                nbpairs.set(atoms.index( int(are13[i][0])), atoms.index(int(are13[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                # print(atoms.index( int(are13[i][0])), atoms.index(int(are13[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                # print (nbpairs)
                # print("~~~~~~~~~~~~~~~~~~`")
                
            for i in range(0, len(are14)): 
                scale_factor1 = 1/2
                scale_factor2 = 1/2
                nbpairs.set(atoms.index( int(are14[i][0])), atoms.index(int(are14[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                mol_params.add14Pair(BondID(atoms.index( int(are14[i][0])), atoms.index( int(are14[i][1]))),scale_factor1 , scale_factor2)
                # print(atoms.index( int(are14[i][0])), atoms.index(int(are14[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                # print (nbpairs)
                # print("~~~~~~~~~~~~~~~~~~`")
                
            for i in range(0, len(nb_pair_list)):  
                scale_factor1 = 1
                scale_factor2 = 1
                nbpairs.set(atoms.index( int(nb_pair_list[i][0])), atoms.index(int(nb_pair_list[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                # print(atoms.index( int(nb_pair_list[i][0])), atoms.index(int(nb_pair_list[i][1])), CLJScaleFactor(scale_factor1,scale_factor2))
                # print (nbpairs)
                # print("~~~~~~~~~~~~~~~~~~`")
                
            mol = editmol.setProperty("intrascale" , nbpairs).commit()
            system.update(mol)


        # print("Setup name of qube FF")
        # mol = mol.edit().setProperty("forcefield", ffToProperty("qube")).commit()
        # system.update(mol)

        molecule = editmol.commit()
        newmolecules.add(molecule)
        return newmolecules




def vsiteListToProperty(list):
    prop = Properties()
    i = 0
    for entry in list:
        for key, value in entry.items():
            prop.setProperty("%s(%d)" % (key,i), VariantProperty(value))
        i += 1
    prop.setProperty("nvirtualsites",VariantProperty(i))
    return prop

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# readCrdTop() returns a "MoleculeGroup", whereas readXmlParameters returns "molecules"

xmlfile = "ethane/MOL.xml"
pdbfile = "ethane/MOL.pdb"
molecules = readXmlParameters(pdbfile, xmlfile)
space = PeriodicBox( Vector(20, 20, 20))
mols =  MoleculeGroup("molecules")
mols.add(molecules)

#(mols, space) = Amber().readCrdTop("ethane/MOL.crd", "ethane/MOL.top")

coul_cutoff = 10 * angstrom
lj_cutoff = 10 * angstrom

rf_diel = 78.3

switchfunc = HarmonicSwitchingFunction(coul_cutoff,coul_cutoff,lj_cutoff,lj_cutoff)

interff = InterCLJFF("interclj")
interff.setProperty("space", space)
interff.setProperty("switchingFunction", switchfunc)
interff.setUseReactionField(True)
interff.setReactionFieldDielectric(rf_diel)
interff.add(mols)

intraclj = IntraCLJFF("intraclj")
intraclj.setProperty("space", space)
intraclj.setProperty("switchingFunction", switchfunc )
intraclj.setUseReactionField(True)
intraclj.setReactionFieldDielectric(rf_diel)
intraclj.add(mols)

intraff = InternalFF("intraff")
intraff.add(mols)

system = System()
system.add(interff)
system.add(intraff)
system.add(intraclj)
system.add(mols)

system.setProperty("space", space)


def _pvt_create_npt_OpenMM(mols, temperature, pressure):
    openmm = OpenMMMDIntegrator(mols)
    openmm.setPlatform("OpenCL")
    openmm.setConstraintType("none")
    openmm.setCutoffType("cutoffperiodic")
    openmm.setIntegrator("leapfrogverlet")
    openmm.setFriction(0.1 * picosecond)
    openmm.setPrecision("double")
    openmm.setTimetoSkip(0*picosecond)
    openmm.setDeviceIndex("0")
    openmm.setLJDispersion(False)
    openmm.setFieldDielectric(rf_diel)
    openmm.setCMMremovalFrequency(0)
    openmm.setBufferFrequency(0)
    openmm.setRestraint(False)
    openmm.setTemperature(temperature)
    openmm.setAndersen(True)
    openmm.setAndersenFrequency(10)
    openmm.setPressure(pressure)
    openmm.setMCBarostat(True)
    openmm.setMCBarostatFrequency(25)
    openmm.initialise()
    return openmm

def _pvt_create_nve_OpenMM(mols):
    openmm = OpenMMMDIntegrator(mols)
    openmm.setPlatform("OpenCL")
    openmm.setConstraintType("none")
    openmm.setCutoffType("cutoffperiodic")
    openmm.setIntegrator("leapfrogverlet")
    openmm.setFriction(0.1 * picosecond)
    openmm.setPrecision("double")
    openmm.setTimetoSkip(0*picosecond)
    openmm.setDeviceIndex("0")
    openmm.setLJDispersion(False)
    openmm.setFieldDielectric(rf_diel)
    openmm.setCMMremovalFrequency(0)
    openmm.setBufferFrequency(0)
    openmm.setRestraint(False)
    openmm.setAndersen(False)
    openmm.setAndersenFrequency(10)
    openmm.setMCBarostat(False)
    openmm.setMCBarostatFrequency(25)
    openmm.initialise()
    return openmm

def _pvt_create_nvt_OpenMM(mols, temperature):
    openmm = OpenMMMDIntegrator(mols)
    openmm.setPlatform("OpenCL")
    openmm.setConstraintType("none")
    openmm.setCutoffType("cutoffperiodic")
    openmm.setIntegrator("leapfrogverlet")
    openmm.setFriction(0.1 * picosecond)
    openmm.setPrecision("double")
    openmm.setTimetoSkip(0*picosecond)
    openmm.setDeviceIndex("0")
    openmm.setLJDispersion(False)
    openmm.setFieldDielectric(rf_diel)
    openmm.setCMMremovalFrequency(0)
    openmm.setBufferFrequency(0)
    openmm.setRestraint(False)
    openmm.setTemperature(temperature)
    openmm.setAndersen(True)
    openmm.setAndersenFrequency(10)
    openmm.setMCBarostat(False)
    openmm.setMCBarostatFrequency(25)
    print("Initialising NVT integrator")
    openmm.initialise()
    return openmm

def test_getters_setters(verbose=False):
    openmm = _pvt_create_nve_OpenMM(system[mols.number()])
    assert (openmm.getAndersen()==False)
    assert (openmm.getPlatform()=="OpenCL")
    assert (openmm.getConstraintType()=="none")
    assert (openmm.getCutoffType()=="cutoffperiodic")
    assert (openmm.getIntegrator() == "leapfrogverlet")
    assert (openmm.getPrecision() == "double")
    assert (openmm.getLJDispersion() == False)


def test_npt_setup(verbose=False):
    if verbose:
        print ("=========NPT energy test============")

    sire_nrg = system.energy().value()

    if verbose:
        print("\nInitial Sire energy = %s kcal mol-1" % sire_nrg)

    openmm = _pvt_create_npt_OpenMM(system[mols.number()], 25*celsius, 1*atm)
    openmm_nrg = openmm.getPotentialEnergy(system).value()

    if verbose:
        print("\nInitial OpenMM energy = %s kcal mol-1" % openmm_nrg)

    assert_almost_equal(sire_nrg, openmm_nrg, 1)

    if verbose:
        print ("========NPT energy test done========")

def test_nvt_setup(verbose=False):
    if verbose:
        print ("=========NVT energy test============")

    sire_nrg = system.energy().value()

    if verbose:
        print("\nInitial Sire energy = %s kcal mol-1" % sire_nrg)

    openmm = _pvt_create_nvt_OpenMM(system[mols.number()], 25*celsius)

    openmm_nrg = openmm.getPotentialEnergy(system).value()

    if verbose:
        print("\nInitial OpenMM energy = %s kcal mol-1" % openmm_nrg)

    assert_almost_equal(sire_nrg, openmm_nrg, 1)

    if verbose:
        print ("========NVT energy test done========")

def test_nve_setup(verbose=False):
    if verbose:
        print ("=========NVE energy test============")

    sire_nrg = system.energy().value()

    if verbose:
        print("\nInitial Sire energy = %s kcal mol-1" % sire_nrg)

    openmm = _pvt_create_nve_OpenMM(system[mols.number()])

    openmm_nrg = openmm.getPotentialEnergy(system).value()

    if verbose:
        print("\nInitial OpenMM energy = %s kcal mol-1" % openmm_nrg)

    assert_almost_equal(sire_nrg, openmm_nrg, 1)

    if verbose:
        print ("========NVE energy test done========")

def _pvt_test_nve(verbose=False):

    openmm = _pvt_create_nve_OpenMM(system[mols.number()])
    openmm_nrg = openmm.getPotentialEnergy(system).value()

    sire_nrg = system.energy().value()
    if verbose:
        print("\nInitial Sire energy = %s kcal mol-1" % sire_nrg)
    if verbose:
        print("\nInitial OpenMM energy = %s kcal mol-1" % openmm_nrg)
    # now let's integrate a few steps
    #system_int = openmm.integrate(system, system.totalComponent(), 2*femtosecond, 10, True)
    #if verbose:
        #print ("Sire Energy after 10 steps of dynamics = %s kcal mol-1" %system_int.energ().value())

    #MD move integration
    velocity_generator = MaxwellBoltzmann(25*celsius)
    velocity_generator.setGenerator(RanGenerator(1234))

    mdmove = MolecularDynamics(mols, openmm, 1*femtosecond,
                              {"velocity generator":velocity_generator})
    tot_e = []
    for i in range(10):
        print("++++++++++++++++++++++++%d++++++++++++++++++++"%i)
        mdmove.move(system, 1, True)
        total_energy = system.energy()+mdmove.kineticEnergy()
        print ("Potential energy is %s and kinetic energy is %s" %(system.energy(),mdmove.kineticEnergy()))
        print ("Total energy from sire is: %s" %total_energy)
        tot_e.append(total_energy.value())
    percent_fluc = np.std(tot_e)/np.abs(np.mean(tot_e))*100
    print ("Percent fluctuations are: %s "%(percent_fluc))
    assert(percent_fluc<0.1)
        #print("Total energy is %s " %total_energy)




if __name__ == "__main__":
    test_npt_setup(True)
    test_nvt_setup(True)
    test_nve_setup(True)
    test_getters_setters(True)
    # #test_nve(True)
