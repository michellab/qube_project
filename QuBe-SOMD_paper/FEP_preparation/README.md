BioSimSpace protocols and scripts used to generate the input files for FEP calculations. 

Step 1 : Use qube_to_prmRst.py to read the xml/pdb files of the protein fragments and save the amber files. Then combine the fragments to gat the whole protein.
Single Point Energy calculations both for the fragments and for the whole protein done with SOMD and OpenMM prove that the file conversion works fine.

Step 2 BSS: Use qube_to_prmRst.py to read the xml/pdb files of the ligands and save the amber files. 

Step 3 BSS: Solvate each ligand

Step 4 BSS: Assemble each protein-ligand complex and solvate

Step 5 BSS Run an MD equilibration for each solvated system

Step 6 BSS: run prepareFEP on every pair of ligands

