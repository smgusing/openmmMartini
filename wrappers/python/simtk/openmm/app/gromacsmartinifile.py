"""
gromacstopfile.py: Used for loading Gromacs top files.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2014 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
author = "Peter Eastman"
# Modified by Gurpreet
version = "1.0"

from simtk.openmm.app import Topology
# from simtk.openmm.app import PDBFile
from . import forcefield as ff
from . import element as elem
from . import amberprmtopfile as prmtop
import simtk.unit as unit
import simtk.openmm as mm
import math
import os
import distutils.spawn

HBonds = ff.HBonds
AllBonds = ff.AllBonds
HAngles = ff.HAngles

OBC2 = prmtop.OBC2


def defaultGromacsIncludeDir():
    """Find the location where gromacs #include files are referenced from, by
    searching for (1) gromacs environment variables, (2) for the gromacs binary
    'pdb2gmx' in the PATH, or (3) just using the default gromacs install
    location, /usr/local/gromacs/share/gromacs/top """
    if 'GMXDATA' in os.environ:
        return os.path.join(os.environ['GMXDATA'], 'gromacs/top')
    if 'GMXBIN' in os.environ:
        return os.path.abspath(
            os.path.join(
                os.environ['GMXBIN'],
                '..',
                'share',
                'gromacs',
                'top'))

    pdb2gmxpath = distutils.spawn.findexecutable('pdb2gmx')
    if pdb2gmxpath is not None:
        return os.path.abspath(
            os.path.join(
                os.path.dirname(pdb2gmxpath),
                '..',
                'share',
                'gromacs',
                'top'))

    return '/usr/local/gromacs/share/gromacs/top'


class GmxtopParser(object):

    """GromacsTopFile parses a Gromacs top infile and constructs a Topology
    and (optionally) an OpenMM System from it."""

    class MoleculeType(object):

        """Inner class to store information about a molecule type."""

        def __init__(self):
            self.atoms = []
            self.bonds = []
            self.angles = []
            self.dihedrals = []
            self.exclusions = []
            self.pairs = []
            self.cmaps = []
            self.constraints = []

    def processFile(self, infile):
        append = ''
        for line in open(infile):
            if line.strip().endswith('\\'):
                append = '%s %s' % (append, line[:line.rfind('\\')])
            else:
                self.processLine(append+' '+line, infile)
                append = ''

    def processLine(self, line, infile):
        """Process one line from a infile."""
        if ';' in line:
            line = line[:line.index(';')]
        stripped = line.strip()
        ignore = not all(self.ifStack)
        if stripped.startswith('*') or len(stripped) == 0:
            # A comment or empty line.
            return

        elif stripped.startswith('[') and not ignore:
            # The start of a category.
            if not stripped.endswith(']'):
                raise ValueError('Illegal line in .top infile: '+line)
            self.currentCategory = stripped[1:-1].strip()
        elif stripped.startswith('#'):
            # A preprocessor command.
            fields = stripped.split()
            command = fields[0]
            if len(self.ifStack) != len(self.elseStack):
                raise RuntimeError('#if/#else stack out of sync')

            if command == '#include' and not ignore:
                # Locate the infile to include
                name = stripped[len(command):].strip(' \t"<>')
                searchDirs = self.includeDirs+(os.path.dirname(infile),)
                for dir in searchDirs:
                    infile = os.path.join(dir, name)
                    if os.path.isfile(infile):
                        # We found the infile, so process it.
                        self.processFile(infile)
                        break
                else:
                    raise ValueError('Could not locate #include infile: '+name)
            elif command == '#define' and not ignore:
                # Add a value to our list of defines.
                if len(fields) < 2:
                    raise ValueError('Illegal line in .top infile: '+line)
                name = fields[1]
                valueStart = stripped.find(name, len(command))+len(name)+1
                value = line[valueStart:].strip()
                self.defines[name] = value
            elif command == '#ifdef':
                # See whether this block should be ignored.
                if len(fields) < 2:
                    raise ValueError('Illegal line in .top infile: '+line)
                name = fields[1]
                self.ifStack.append(name in self.defines)
                self.elseStack.append(False)
            elif command == '#ifndef':
                # See whether this block should be ignored.
                if len(fields) < 2:
                    raise ValueError('Illegal line in .top infile: '+line)
                name = fields[1]
                self.ifStack.append(name not in self.defines)
                self.elseStack.append(False)
            elif command == '#endif':
                # Pop an entry off the if stack.
                if len(self.ifStack) == 0:
                    raise ValueError('Unexpected line in .top file: '+line)
                del(self.ifStack[-1])
                del(self.elseStack[-1])
            elif command == '#else':
                # Reverse the last entry on the if stack
                if len(self.ifStack) == 0:
                    raise ValueError('Unexpected line in .top file: '+line)
                if self.elseStack[-1]:
                    raise ValueError('Unexpected line in .top file: '
                                     '#else has already been used ' + line)
                self.ifStack[-1] = (not self.ifStack[-1])
                self.elseStack[-1] = True

        elif not ignore:
            # A line of data for the current category
            if self.currentCategory is None:
                raise ValueError('Unexpected line in .top file: '+line)
            if self.currentCategory == 'defaults':
                self.processDefaults(line)
            elif self.currentCategory == 'nonbond_params':
                self.processNonbond_params(line)
            elif self.currentCategory == 'moleculetype':
                self.processMoleculeType(line)
            elif self.currentCategory == 'molecules':
                self.processMolecule(line)
            elif self.currentCategory == 'atoms':
                self.processAtom(line)
            elif self.currentCategory == 'bonds':
                self.processBond(line)
            elif self.currentCategory == 'constraints':
                self.processConstraints(line)
            elif self.currentCategory == 'angles':
                self.processAngle(line)
            elif self.currentCategory == 'dihedrals':
                self.processDihedral(line)
            elif self.currentCategory == 'exclusions':
                self.processExclusion(line)
            elif self.currentCategory == 'pairs':
                self.processPair(line)
            elif self.currentCategory == 'cmap':
                self.processCmap(line)
            elif self.currentCategory == 'atomtypes':
                self.processAtomType(line)
            elif self.currentCategory == 'bondtypes':
                self.processBondType(line)
            elif self.currentCategory == 'angletypes':
                self.processAngleType(line)
            elif self.currentCategory == 'dihedraltypes':
                self.processDihedralType(line)
            elif self.currentCategory == 'implicitgenbornparams':
                self.processImplicitType(line)
            elif self.currentCategory == 'pairtypes':
                self.processPairType(line)
            elif self.currentCategory == 'cmaptypes':
                self.processCmapType(line)

    def processDefaults(self, line):
        """Process the [ defaults ] line."""
        fields = line.split()
        if len(fields) < 4:
            raise ValueError('Too few fields in [ defaults ] line: '+line)
        if fields[0] != '1':
            raise ValueError('Unsupported nonbonded type: '+fields[0])
        if fields[1] != '2':
            raise ValueError('Unsupported combination rule: '+fields[1])
        if fields[2].lower() == 'no':
            raise ValueError('genpairs=no is not supported')
        self.defaults = fields
    
    def processNonbond_params(self, line):
        """Process [ nonbond_params ] category"""
        fields = line.split()
        if len(fields) < 4:
            msg = "Too few fields in nonbond_params %s"%fields
            raise ValueError(msg)
        else:
            pass
        sig,eps = self.convertVWtoSigEps(fields[3],fields[4])
        self.nonbondParams[tuple(fields[:2])] = (sig,eps)
        

    def processMoleculeType(self, line):
        """Process a line in the [ moleculetypes ] category."""
        fields = line.split()
        if len(fields) < 1:
            raise ValueError('Too few fields in [ moleculetypes ] line: '+line)
        moltype = self.MoleculeType()
        self.moleculeTypes[fields[0]] = moltype
        self.currentMoleculeType = moltype

    def processMolecule(self, line):
        """Process a line in the [ molecules ] category."""
        fields = line.split()
        if len(fields) < 2:
            raise ValueError('Too few fields in [ molecules ] line: '+line)
        self.molecules.append((fields[0], int(fields[1])))

    def processAtom(self, line):
        """Process a line in the [ atoms ] category."""
        if self.currentMoleculeType is None:
            raise ValueError('Found [ atoms ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ atoms ] line: '+line)
        self.currentMoleculeType.atoms.append(fields)

    def processBond(self, line):
        """Process a line in the [ bonds ] category."""
        if self.currentMoleculeType is None:
            raise ValueError('Found [ bonds ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 3:
            raise ValueError('Too few fields in [ bonds ] line: '+line)
        if fields[2] != '1':
            raise ValueError(
                'Unsupported function type in [ bonds ] line: ' +
                line)
        self.currentMoleculeType.bonds.append(fields)
        
    def processConstraints(self, line):
        """Process a line in the [ constraints ] category."""
        if self.currentMoleculeType is None:
            raise ValueError('Found [ constraints ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 3:
            raise ValueError('Too few fields in [ constraints ] line: '+line)
        if fields[2] != '1':
            raise ValueError(
                'Unsupported function type in [ constraints ] line: ' +
                line)
        self.currentMoleculeType.constraints.append(fields)


    def processAngle(self, line):
        """Process a line in the [ angles ] category."""
        if self.currentMoleculeType is None:
            raise ValueError('Found [ angles ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 4:
            raise ValueError('Too few fields in [ angles ] line: '+line)
        if fields[3] not in ('1', '5'):
            raise ValueError(
                'Unsupported function type in [ angles ] line: ' +
                line)
        self.currentMoleculeType.angles.append(fields)

    def processDihedral(self, line):
        """Process a line in the [ dihedrals ] category."""
        if self.currentMoleculeType is None:
            raise ValueError(
                'Found [ dihedrals ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ dihedrals ] line: '+line)
        if fields[4] not in ('1', '2', '3', '4', '9'):
            raise ValueError(
                'Unsupported function type in [ dihedrals ] line: ' +
                line)
        self.currentMoleculeType.dihedrals.append(fields)

    def processExclusion(self, line):
        """Process a line in the [ exclusions ] category."""
        if self.currentMoleculeType is None:
            raise ValueError(
                'Found [ exclusions ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 2:
            raise ValueError('Too few fields in [ exclusions ] line: '+line)
        self.currentMoleculeType.exclusions.append(fields)

    def processPair(self, line):
        """Process a line in the [ pairs ] category."""
        if self.currentMoleculeType is None:
            raise ValueError('Found [ pairs ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 3:
            raise ValueError('Too few fields in [ pairs ] line: '+line)
        if fields[2] != '1':
            raise ValueError(
                'Unsupported function type in [ pairs ] line: ' +
                line)
        self.currentMoleculeType.pairs.append(fields)

    def processCmap(self, line):
        """Process a line in the [ cmaps ] category."""
        if self.currentMoleculeType is None:
            raise ValueError('Found [ cmap ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('Too few fields in [ pairs ] line: '+line)
        self.currentMoleculeType.cmaps.append(fields)

    def processAtomType(self, line):
        """Process a line in the [ atomtypes ] category."""
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('Too few fields in [ atomtypes ] line: '+line)
        if len(fields[3]) == 1:
            # Bonded type and atomic number are both missing.
            fields.insert(1, None)
            fields.insert(1, None)
        elif len(fields[4]) == 1 and len(fields[5]) > 1:
            if fields[1][0].isalpha():
                # Atomic number is missing.
                fields.insert(2, None)
            else:
                # Bonded type is missing.
                fields.insert(1, None)
        self.atomTypes[fields[0]] = fields

    def processBondType(self, line):
        """Process a line in the [ bondtypes ] category."""
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ bondtypes ] line: '+line)
        if fields[2] != '1':
            raise ValueError(
                'Unsupported function type in [ bondtypes ] line: ' +
                line)
        self.bondTypes[tuple(fields[:2])] = fields

    def processAngleType(self, line):
        """Process a line in the [ angletypes ] category."""
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('Too few fields in [ angletypes ] line: '+line)
        if fields[3] not in ('1', '5'):
            raise ValueError(
                'Unsupported function type in [ angletypes ] line: ' +
                line)
        self.angleTypes[tuple(fields[:3])] = fields

    def processDihedralType(self, line):
        """Process a line in the [ dihedraltypes ] category."""
        fields = line.split()
        if len(fields) < 7:
            raise ValueError('Too few fields in [ dihedraltypes ] line: '+line)
        if fields[4] not in ('1', '2', '3', '4', '9'):
            raise ValueError(
                'Unsupported function type in [ dihedraltypes ] line: ' +
                line)
        key = tuple(fields[:5])
        if fields[4] == '9' and key in self.dihedralTypes:
            # There are multiple dihedrals defined for these atom types.
            self.dihedralTypes[key].append(fields)
        else:
            self.dihedralTypes[key] = [fields]

    def processImplicitType(self, line):
        """Process a line in the [ implicitgenbornparams ] category."""
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(
                'Too few fields in [ implicitgenbornparams ] line: ' +
                line)
        self.implicitTypes[fields[0]] = fields

    def processPairType(self, line):
        """Process a line in the [ pairtypes ] category."""
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ pairtypes] line: '+line)
        if fields[2] != '1':
            raise ValueError(
                'Unsupported function type in [ pairtypes ] line: ' +
                line)
        self.pairTypes[tuple(fields[:2])] = fields

    def processCmapType(self, line):
        """Process a line in the [ cmaptypes ] category."""
        fields = line.split()
        if len(fields) < 8 or len(fields) < 8+int(fields[6])*int(fields[7]):
            raise ValueError('Too few fields in [ cmaptypes ] line: '+line)
        if fields[5] != '1':
            raise ValueError(
                'Unsupported function type in [ cmaptypes ] line: ' +
                line)
        self.cmapTypes[tuple(fields[:5])] = fields

    def convertVWtoSigEps(self, v, w):
        '''convert c6, c12 to sig, eps'''
        v = float(v)
        w = float(w)
        try:
            #sig = round( (w/v) ** (1.0/6.0), 3 )
            #eps = round( (v * v)/ (4.0 * w), 3 )
            sig = round( (w/v) ** (1.0/6.0), 5 )
            eps = round( (v * v)/ (4.0 * w), 5 )
        except ZeroDivisionError:
            print("Divide by zero occured. Will set both sig,eps to zero")
            sig, eps = 0.0, 0.0
            
        return sig, eps
                          
        
    def __init__(
            self,
            infile,
            unitCellDimensions=None,
            includeDir=None,
            defines=None):
        """Load a top file.

        Parameters:
         - infile (string) the name of the infile to load
         - unitCellDimensions (Vec3=None) the dimensions of the crystallographic unit cell
         - includeDir (string=None) A directory in which to look for other files
           included from the top infile. If not specified, we will attempt to locate a gromacs
           installation on your system. When gromacs is installed in /usr/local, this will resolve
           to  /usr/local/gromacs/share/gromacs/top
         - defines (dict={}) preprocessor definitions that should be predefined when parsing the infile
         """
        if includeDir is None:
            includeDir = defaultGromacsIncludeDir()
        self.includeDirs = (os.path.dirname(infile), includeDir)
        # Most of the gromacs water itp files for different forcefields,
        # unless the preprocessor #define FLEXIBLE is given, don't define
        # bonds between the water hydrogen and oxygens, but only give the
        # constraint distances and exclusions.
        self.topfile = infile
        self.defines = {'FLEXIBLE': True}
        if defines is not None:
            self.defines.update(defines)
        # Parse the infile.
        self.currentCategory = None
        self.ifStack = []
        self.elseStack = []
        self.nonbondParams = {}
        self.moleculeTypes = {}
        self.molecules = []
        self.currentMoleculeType = None
        self.atomTypes = {}
        self.bondTypes = {}
        self.angleTypes = {}
        self.dihedralTypes = {}
        self.implicitTypes = {}
        self.pairTypes = {}
        self.cmapTypes = {}


class GmxTopMartini(GmxtopParser):

    """ Parses gromacs topology files pertaining to MARTINI Force field."""

    def __init__(
            self,
            infile,
            unitCellDimensions=None,
            includeDir=None,
            defines=None):
        """Load a top infile.

        Parameters:
         - infile (string) the name of the infile to load
         - unitCellDimensions (Vec3=None) the dimensions of the crystallographic unit cell
         - includeDir (string=None) A directory in which to look for other files
           included from the top infile. If not specified, we will attempt to locate a gromacs
           installation on your system. When gromacs is installed in /usr/local, this will resolve
           to  /usr/local/gromacs/share/gromacs/top
         - defines (dict={}) preprocessor definitions that should be predefined when parsing the infile
         """

        super(GmxTopMartini, self).__init__(infile,
                                            unitCellDimensions,
                                            includeDir,
                                            defines)

        # Overwrite methods in base class that do not work for MARTINI topology
        self.infile = infile
        self.unitCellDimensions = unitCellDimensions
        # self.processFile(infile)
        self.genTop()

    def processDefaults(self, line):
        """Process the [ defaults ] line."""

        fields = line.split()
        if len(fields) > 2:
            raise ValueError('Too many fields in [ defaults ] line: '+line)
        if fields[0] != '1':
            raise ValueError('Unsupported nonbonded type: '+fields[0])
        if fields[1] != '1':
            raise ValueError('Unsupported combination rule: '+fields[1])
        self.defaults = fields

    def processAngle(self, line):
        """Process a line in the [ angles ] category."""
        if self.currentMoleculeType is None:
            raise ValueError('Found [ angles ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 4:
            raise ValueError('Too few fields in [ angles ] line: '+line)
        if fields[3] not in ('2'):
            raise ValueError(
                'Unsupported (nonG96) function type in [ angles ] line: ' +
                line)
        self.currentMoleculeType.angles.append(fields)

    def genTop(self):
        self.processFile(self.infile)
        # Create the Topology from it.
        top = Topology()
        # The Topology read from the prmtop infile
        self.topology = top
        top.setUnitCellDimensions(self.unitCellDimensions)
        # PDBFile._loadNameReplacementTables()
        for moleculeName, moleculeCount in self.molecules:
            if moleculeName not in self.moleculeTypes:
                raise ValueError("Unknown molecule type: "+moleculeName)
            moleculeType = self.moleculeTypes[moleculeName]

            # Create the specified number of molecules of this type.

            for i in range(moleculeCount):
                atoms = []
                lastResidue = None
                c = top.addChain()
                for index, fields in enumerate(moleculeType.atoms):
                    resNumber = fields[2]
                    if resNumber != lastResidue:
                        lastResidue = resNumber
                        resName = fields[3]
                        r = top.addResidue(resName, c)
                    atomName = fields[4]
                    atoms.append(top.addAtom(atomName,None, r))

                # Add bonds to the topology
                for fields in moleculeType.bonds:
                    top.addBond(
                        atoms[int(fields[0])-1], atoms[int(fields[1])-1])

    def createSystem(
            self,
            nonbondedMethod=None,
            nonbondedCutoff=None,
            removeCMMotion=True,
            constraints = None            
            ):
        """Construct an OpenMM System representing the topology described by
        this prmtop infile.

        Parameters:
         - nonbondedMethod (object=NoCutoff) The method to use for nonbonded
         interactions.  Allowed values are
           NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, or PME.
         - nonbondedCutoff (distance=1*nanometer) The cutoff distance to use for
         nonbonded interactions
         - constraints (object=None) Specifies which bonds and angles should be
         implemented with constraints. Allowed values are None, HBonds, AllBonds
         or HAngles.
         - rigidWater (boolean=True) If true, water molecules will be fully
         rigid regardless of the value passed for the constraints argument
         - implicitSolvent (object=None) If not None, the implicit solvent model
         to use.  The only allowed value is OBC2.
         - soluteDielectric (float=1.0) The solute dielectric constant to use in
         the implicit solvent model.
         - solventDielectric (float=78.5) The solvent dielectric constant to use
         in the implicit solvent model.
         - ewaldErrorTolerance (float=0.0005) The error tolerance to use if
         nonbondedMethod is Ewald or PME.
         - removeCMMotion (boolean=True) If true, a CMMotionRemover will be
         added to the System
         - hydrogenMass (mass=None) The mass to use for hydrogen atoms bound to
         heavy atoms.  Any mass added to a hydrogen is
           subtracted from the heavy atom to keep their total mass the same.
        Returns: the newly created System
        """
        # Create the System.

        syst = mm.System()
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(nonbondedMethod)
        nb.setCutoffDistance(nonbondedCutoff)
        syst.addForce(nb)
      
        boxSize = self.topology.getUnitCellDimensions()
        if boxSize is not None:
            syst.setDefaultPeriodicBoxVectors(
                (boxSize[0], 0, 0), (0, boxSize[1], 0), (0, 0, boxSize[2]))
            
            
        # Build a lookup table to let us process dihedrals more quickly.
        dihedralTypeTable, wildcardDihedralTypes = self._buildDihLookupTable()

        # Loop over molecules and create the specified number of each type.
        allAtomTypes = []
        allcharges = []
        allExceptions = [] 
        
        
         
        for moleculeName, moleculeCount in self.molecules:
            moleculeType = self.moleculeTypes[moleculeName]
            for i in range(moleculeCount):
                # Record the types of all atoms.
                baseAtomIndex = syst.getNumParticles()
                atomTypes = [atom[1] for atom in moleculeType.atoms]
                charges = [atom[6] for atom in moleculeType.atoms ]
                for charge in charges: allcharges.append(charge)
                for atomType in atomTypes: allAtomTypes.append(atomType)
                
                try:
                    bondedTypes = [self.atomTypes[t][1] for t in atomTypes]
                except KeyError as e:
                    raise ValueError('Unknown atom type: '+e.message)
                bondedTypes = [
                    b if b is not None else a for a,
                    b in zip(
                        atomTypes,
                        bondedTypes)]
                # Add atoms.
                self._addAtomsToSystem(syst, moleculeType)

                # Add bonds.
                atomBonds = self._addBondsToSystem(syst, moleculeType, bondedTypes, 
                                       constraints, baseAtomIndex)
                # Add constraints
                self._addConstraintsToSystem(syst, moleculeType, bondedTypes, 
                                       constraints, baseAtomIndex)

                # Add angles.
                self._addAngleToSystem(syst, moleculeType, bondedTypes, atomBonds,
                                       baseAtomIndex)

                # Add torsions.
                self._addTorsionToSystem(syst, moleculeType, bondedTypes,
                            dihedralTypeTable, wildcardDihedralTypes, baseAtomIndex)                
                
                # Set nonbonded parameters for particles.
                exceptions = self._setnonbondedParams(nb, moleculeType, baseAtomIndex, atomTypes)
                for exception in exceptions: allExceptions.append(exception)
                 
        # Add pairInteractions first as exceptions, followed by the rest
        # This way other exceptions can override pairInteractions
        for i in range(syst.getNumParticles()-1 ):
            atomType1 = allAtomTypes[i]
            for j in range(i+1, syst.getNumParticles()):
                atomType2 = allAtomTypes[j]
                
                try:
                    sig,eps = self.nonbondParams[(atomType1,atomType2)]
                except KeyError:
                    try:
                        sig,eps = self.nonbondParams[(atomType2,atomType1)]
                    except KeyError():
                        msg = "%s,%s pair interactions not found"%(atomType2,atomType1)
                        raise KeyError(msg)
                
                chargeProd = float(allcharges[i]) * float(allcharges[j])
                nb.addException(i, j, chargeProd, sig, eps, True)
                
        
        for exception in allExceptions:
            nb.addException( exception[0], exception[1], exception[2], 
                       float(exception[3]), float(exception[4]), True)
        
 
        # Add a CMMotionRemover.
        if removeCMMotion:
            syst.addForce(mm.CMMotionRemover())
        
        return syst
                    

        
    def _buildDihLookupTable(self):

        dihedralTypeTable = {}
        for key in self.dihedralTypes:
            if key[1] != 'X' and key[2] != 'X':
                if (key[1], key[2]) not in dihedralTypeTable:
                    dihedralTypeTable[(key[1], key[2])] = []
                dihedralTypeTable[(key[1], key[2])].append(key)
                if (key[2], key[1]) not in dihedralTypeTable:
                    dihedralTypeTable[(key[2], key[1])] = []
                dihedralTypeTable[(key[2], key[1])].append(key)
        wildcardDihedralTypes = []
        for key in self.dihedralTypes:
            if key[1] == 'X' or key[2] == 'X':
                wildcardDihedralTypes.append(key)
                for types in dihedralTypeTable.itervalues():
                    types.append(key)
        
        return dihedralTypeTable, wildcardDihedralTypes
    
    def _addAtomsToSystem(self, syst, moleculeType):
        
        for fields in moleculeType.atoms:
            if len(fields) >= 8:
                mass = float(fields[7])
            else:
                mass = float(self.atomTypes[fields[1]][3])
            syst.addParticle(mass)
            
    def _addBondsToSystem(self, syst, moleculeType, bondedTypes,
                          constraints, baseAtomIndex):
        
        atomBonds = [{} for x in range(len(moleculeType.atoms))]
        for fields in moleculeType.bonds:
            atoms = [int(x)-1 for x in fields[:2]]
            types = tuple(bondedTypes[i] for i in atoms)
            if len(fields) >= 5:
                params = fields[3:5]
            elif types in self.bondTypes:
                params = self.bondTypes[types][3:5]
            elif types[::-1] in self.bondTypes:
                params = self.bondTypes[types[::-1]][3:5]
            else:
                raise ValueError(
                    'No parameters specified for bond: ' +
                    fields[0] +
                    ', ' +
                    fields[1])
            # Decide whether to use a constraint or a bond.
            useConstraint = False
            if constraints is AllBonds:
                useConstraint = True
            # Add the bond or constraint.
            length = float(params[0])
            if useConstraint:
                syst.addConstraint(baseAtomIndex +  atoms[0],
                    baseAtomIndex + atoms[1],    length)
            else:
                bonds = mm.HarmonicBondForce()
                syst.addForce(bonds)
                bonds.addBond( baseAtomIndex + atoms[0],
                               baseAtomIndex + atoms[1],
                               length, float(params[1]))
            
            # Record information that will be needed for constraining
            # angles.
            atomBonds[atoms[0]][atoms[1]] = length
            atomBonds[atoms[1]][atoms[0]] = length
        return atomBonds

    def _addConstraintsToSystem(self, syst, moleculeType, bondedTypes,
                          constraints, baseAtomIndex):
        
        for fields in moleculeType.constraints:
            atoms = [int(x)-1 for x in fields[:2]]
            length = float(fields[3])
            syst.addConstraint(baseAtomIndex + atoms[0],
                    baseAtomIndex + atoms[1], length)

        
    def _addAngleToSystem(self, syst, moleculeType, bondedTypes, atomBonds,
                           baseAtomIndex):
        
        degToRad = math.pi/180
        for fields in moleculeType.angles:
            atoms = [int(x)-1 for x in fields[:3]]
            types = tuple(bondedTypes[i] for i in atoms)
            if len(fields) >= 6:
                params = fields[4:]
            elif types in self.angleTypes:
                params = self.angleTypes[types][4:]
            elif types[::-1] in self.angleTypes:
                params = self.angleTypes[types[::-1]][4:]
            else:
                raise ValueError(
                    'No parameters specified for angle: ' +
                    fields[0] + ', ' + fields[1] + ', ' + fields[2])
                
            #angles = mm.HarmonicAngleForce()
            theta = float(params[0])*degToRad
            if int(fields[3]) == 2: 
                gromosAngle = mm.CustomAngleForce('0.5*k*(cos(theta)-cos(theta0))^2')
                gromosAngle.addPerAngleParameter('theta0')
                gromosAngle.addPerAngleParameter('k')
                syst.addForce(gromosAngle)
                gromosAngle.addAngle(baseAtomIndex +  atoms[0],
                    baseAtomIndex + atoms[1],
                    baseAtomIndex + atoms[2],
                    [theta, float(params[1])])
                
            elif int(fields[3]) == 1:
                angles = mm.HarmonicAngleForce()
                angles.addAngle(baseAtomIndex +  atoms[0],
                                 baseAtomIndex + atoms[1],
                                 baseAtomIndex + atoms[2],
                                 theta, float(params[1]))

            
            
    def _addTorsionToSystem(self, syst, moleculeType, bondedTypes,
                            dihedralTypeTable, wildcardDihedralTypes, baseAtomIndex):
            
        degToRad = math.pi/180
        for fields in moleculeType.dihedrals:
            atoms = [int(x)-1 for x in fields[:4]]
            types = tuple(bondedTypes[i] for i in atoms)
            dihedralType = fields[4]
            reversedTypes = types[::-1]+(dihedralType,)
            types = types+(dihedralType,)
            if (dihedralType in ('1', '2', '4', '9') and len(fields) > 6) or (dihedralType == '3' and len(fields) > 10):
                paramsList = [fields]
            else:
                # Look for a matching dihedral type.
                paramsList = None
                if (types[1], types[2]) in dihedralTypeTable:
                    dihedralTypes = dihedralTypeTable[
                        (types[1], types[2])]
                else:
                    dihedralTypes = wildcardDihedralTypes
                for key in dihedralTypes:
                    if all(a == b or a == 'X' for a, b in zip(key, types)) or all(a == b or a == 'X' for a, b in zip(key, reversedTypes)):
                        paramsList = self.dihedralTypes[key]
                        if 'X' not in key:
                            break
                if paramsList is None:
                    raise ValueError(
                        'No parameters specified for dihedral: ' +
                        fields[0] +  ', ' + fields[1] + ', ' + fields[2] + ', ' + fields[3])
                    
            for params in paramsList:
                if dihedralType in ('1', '4', '9'):
                    # Periodic torsion
                    k = float(params[6])
                    if k != 0:
                        periodic = mm.PeriodicTorsionForce()
                        syst.addForce(periodic)
                        periodic.addTorsion(baseAtomIndex + atoms[0], 
                            baseAtomIndex + atoms[1], baseAtomIndex + atoms[2],
                            baseAtomIndex + atoms[3], int(params[7]),
                            float(params[5]) * degToRad, k)
                        
                elif dihedralType == '2':
                    # Harmonic torsion
                    k = float(params[6])
                    if k != 0:
                        harmonicTorsion = mm.CustomTorsionForce('0.5*k*(theta-theta0)^2')
                        harmonicTorsion.addPerTorsionParameter('theta0')
                        harmonicTorsion.addPerTorsionParameter('k')
                        syst.addForce(harmonicTorsion)
                        harmonicTorsion.addTorsion(baseAtomIndex + atoms[0], baseAtomIndex +
                            atoms[1], baseAtomIndex + atoms[2],
                            baseAtomIndex + atoms[3], (float(params[5]) * degToRad, k))
                else:
                    # RB Torsion
                    c = [float(x) for x in params[5:11]]
                    if any(x != 0 for x in c):
                        rb = mm.RBTorsionForce()
                        syst.addForce(rb)
                        rb.addTorsion( 
                            baseAtomIndex +  atoms[0],
                            baseAtomIndex + atoms[1],
                            baseAtomIndex + atoms[2],
                            baseAtomIndex + atoms[3],
                            c[0], c[1], c[2], c[3], c[4],c[5])
                        

    def _setnonbondedParams(self, nb, moleculeType, baseAtomIndex, atomTypes):

        
        for fields in moleculeType.atoms:
            params = self.atomTypes[fields[1]]
            if len(fields) > 6:
                q = float(fields[6])
            else:
                q = float(params[4])
            nb.addParticle(q, float(params[6]), float(params[7]))
        

        pairExceptions = self._genPairExceptions(nb, moleculeType, baseAtomIndex, atomTypes)
        exclusionExceptions = self._genExclusionExceptions(nb, moleculeType, baseAtomIndex, atomTypes)
        bondedExceptions = self._genBondedExceptions(moleculeType, baseAtomIndex)
        constraintExceptions = self._genConstraintExceptions(moleculeType, baseAtomIndex)
        # order matters
        exceptions =  pairExceptions + exclusionExceptions + bondedExceptions + constraintExceptions         

        return exceptions

    def _genPairExceptions(self, nb, moleculeType, baseAtomIndex, atomTypes):
        
        pairExceptions = []
        for fields in moleculeType.pairs:
            atoms = [int(x)-1 for x in fields[:2]]
            types = tuple(atomTypes[i] for i in atoms)
            if len(fields) >= 5:
                params = fields[3:5]
            elif types in self.pairTypes:
                params = self.pairTypes[types][3:5]
            elif types[::-1] in self.pairTypes:
                params = self.pairTypes[types[::-1]][3:5]
            else:
                # We'll use the automatically generated parameters
                continue
            
            atom1params = nb.getParticleParameters( baseAtomIndex+atoms[0])
            atom2params = nb.getParticleParameters(baseAtomIndex+atoms[1])
            pairExceptions.append(
                    (baseAtomIndex+atoms[0],
                    baseAtomIndex+atoms[1], 
                    atom1params[0]*atom2params[0],
                    params[0], params[1]))
            
        return pairExceptions

    def _genExclusionExceptions(self, nb, moleculeType, baseAtomIndex, atomTypes):

        exclusionExceptions = []
        for fields in moleculeType.exclusions:
            atoms = [int(x)-1 for x in fields]
            for atom in atoms[1:]:
                if atom > atoms[0]:
                    exclusionExceptions.append(
                        (baseAtomIndex+atoms[0], baseAtomIndex+atom, 0, 0, 0))
        return exclusionExceptions
    
    def _genBondedExceptions(self, moleculeType, baseAtomIndex):
        
        bondIndices = []
        for fields in moleculeType.bonds:
            atoms = [int(x)-1 for x in fields[:2]]
            bondIndices.append(
                (baseAtomIndex+atoms[0], baseAtomIndex+atoms[1]))
        
        bondedExceptions = [ (i, j, 0, 0, 0) for i,j in bondIndices ]
        return bondedExceptions
        
    def _genConstraintExceptions(self, moleculeType, baseAtomIndex):
        
        constraintIndices = []
        for fields in moleculeType.constraints:
            atoms = [int(x)-1 for x in fields[:2]]
            constraintIndices.append(
                (baseAtomIndex+atoms[0], baseAtomIndex+atoms[1]))
        
        constraintExceptions = [ (i, j, 0, 0, 0) for i,j in constraintIndices ]
        return constraintExceptions
    
