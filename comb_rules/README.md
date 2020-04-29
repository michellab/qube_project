Modified files: 
- corelib/src/lib/SireMove/openmmfrenergyst.*
- wrapper/Move/OpenMMFrenergyST.pypp.cpp

The setupMovesFreeEnergy() function should also have this line: 
Integrator_OpenMM.setCombiningRules(combining_rules.val)
Typically this is at the OpenMMMD.py file at wrapper/Tools, but here I just include the function at the test.py file
