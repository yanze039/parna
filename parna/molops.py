import numpy as np
import rdkit
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import Conformer
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
# BUG: only when this import exists, UFFGetMoleculeForceField can be registered.
from rdkit.Chem import ChemicalForceFields
from parna.logger import getLogger
from parna.utils import map_atoms, normalize, split_xyz_file
from parna.qm.xtb import XTB
from typing import List, Tuple
import os
from pathlib import Path
from parna.parm import parameterize, generate_frcmod
from parna.resp import RESP_fragment
import parmed as pmd
import openmm as mm
import openmm.app as app


logger = getLogger(__name__)


def flexible_align(mobile, template, atom_mapping=None, force_constant=100.0, max_iterations=6):
    if atom_mapping is None:
        atom_mapping = map_atoms(mobile, template)
    _ = rdMolAlign.AlignMol(mobile, template, atomMap=atom_mapping)
    ff = UFFGetMoleculeForceField(mobile)
    template_conf = template.GetConformer(0)
    for i_q, i_t in atom_mapping:
        p_t = template_conf.GetAtomPosition(i_t)
        pIdx = ff.AddExtraPoint(p_t.x, p_t.y, p_t.z, fixed=True) -1
        ff.UFFAddDistanceConstraint(pIdx, i_q, False, 0., 0., force_constant)
    ff.Initialize()
    logger.info("Constrained Energy minimization...")
    for it in range(max_iterations):
        logger.info(f"Minimization iteration {it+1}")
        more = ff.Minimize(maxIts=20000, energyTol=1e-4, forceTol=1e-3)
        if int(more) == 0:
            logger.info("Converged!")
            break

    return mobile


def get_bond_vector(mol, index1, index2):
    pos1 = mol.GetConformer().GetAtomPosition(index1)
    pos2 = mol.GetConformer().GetAtomPosition(index2)
    return np.array(pos2) - np.array(pos1)


def get_norm_vector(mol, center, ter1, ter2):
    bvec1 = get_bond_vector(mol, center, ter1)
    bvec2 = get_bond_vector(mol, center, ter2)
    norm_bvec = np.cross(bvec1, bvec2)
    norm_bvec = norm_bvec / np.linalg.norm(norm_bvec)
    return norm_bvec


def ortho_frame(vec1, vec2):
    x = normalize(vec1)
    y = normalize(np.cross(vec1, vec2))
    z = normalize(np.cross(x, y))
    return np.stack([x,y,z])


def construct_local_frame(mol, center, ringatom1, ringatom2, sugeratom):
    """Construct the local frame of the base. represented by a triangle in 3d space.
        Args:
            center: atom index of the nitrogen center;
            ringatom1: atom index of first atom on the base ring;
            ringatom2: atom index of second atom on the base ring;
            sugeratom: atom index of the C1' on suger.
    
    """
    posc = mol.GetConformer().GetAtomPosition(center)
    pos1 = mol.GetConformer().GetAtomPosition(ringatom1)
    pos2 = mol.GetConformer().GetAtomPosition(ringatom2)
    poss = mol.GetConformer().GetAtomPosition(sugeratom)
    # one edge is just the bond pointing from nitrogen to suger
    vector1 = normalize(np.array(poss) - np.array(posc))
    # second edge is the normal of the plane
    bvec1 = np.array(pos1) - np.array(posc)
    bvec2 = np.array(pos2) - np.array(posc)
    norm_bvec = np.cross(bvec1, bvec2)
    norm_bvec = norm_bvec / np.linalg.norm(norm_bvec)
    return vector1, norm_bvec


def rotate_conformer(conformer: Conformer, matrix: np.ndarray):
    """
    Apply a 3x3 transformational matrix to the coordinates of the conformer.

    This is a convenience function that fits the 3x3 matrix into a 4x4 matrix to
    include the extra translational dimension required by `TransformConformer`.

    :param conformer: a conformer with nonzero coordinates
    :param matrix: a 3x3 matrix to transform the conformer coordinates
    """
    mat_with_translation = np.identity(4)
    for idx, col in enumerate(matrix):
        mat_with_translation[idx][:3] = col
    rdMolTransforms.TransformConformer(conformer, mat_with_translation)


def translate_conformer(conformer: Conformer, matrix: np.ndarray):
    """
    Apply a 3x3 transformational matrix to the coordinates of the conformer.

    This is a convenience function that fits the 3x3 matrix into a 4x4 matrix to
    include the extra translational dimension required by `TransformConformer`.

    :param conformer: a conformer with nonzero coordinates
    :param matrix: a 3x3 matrix to transform the conformer coordinates
    """
    mat_with_translation = np.identity(4)
    mat_with_translation[:3, 3] = matrix
    rdMolTransforms.TransformConformer(conformer, mat_with_translation)


class MoleculeFactory:
    def __init__(
            self,
            mol_name: str = "mol",
            atom_type: str = "amber",
            threads: int = 48,
            memory: str = "160 GB",
            resp_method: str = "HF",
            resp_basis: str = "6-31G*",
            resp_n_conformers: int = 6,
            output_dir: str = None
        ):
        self.threads = threads
        self.memory = memory
        self.resp_method = resp_method
        self.resp_basis = resp_basis
        self.resp_n_conformers = resp_n_conformers
        self.mol_name = mol_name
        self.atom_type = atom_type
        self.output_dir = output_dir
    
    def charge_molecule(self, input_file, charge, output_dir):
        """
        Calculate RESP charges for a molecule.
        Input:
            input_file: str, `PDB` or `XYZ` file.
            charge: int, charge of the molecule
            output_dir: str, path to the output directory
        """
        RESP_fragment(
            str(input_file),
            charge,
            str(output_dir),
            self.mol_name,
            memory=self.memory, 
            n_threads=self.threads, 
            method_basis=f"{self.resp_method}/{self.resp_basis}",
            n_conformers=self.resp_n_conformers
        )
    
    def parameterize(self, pdb_file, output_dir, prefix, addons=[]):
        """
        Parameterize a molecule using antechamber and tleap.
        Input:
            pdb_file: str, `PDB` file
            output_dir: str, path to the output directory
        """
        output_dir = Path(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pdb_file = Path(pdb_file)
        tmp_mol2 = output_dir / "tmp.mol2"
        command_antechamber = [
            "antechamber",
            "-fi", "pdb",
            "-i", str(pdb_file),
            "-fo", "mol2",
            "-o", str(tmp_mol2),
            "-at", self.atom_type,
            "-pf", "y",
            "-rn", self.mol_name,
        ]
        os.system(" ".join(command_antechamber))
        generate_frcmod(
            input_file=str(tmp_mol2),
            output_file=str(output_dir/f"{pdb_file.stem}.frcmod"),
            sinitize=False
        )
        mol2_pmd = pmd.load_file(str(tmp_mol2))
        for atom in mol2_pmd.atoms:
            atom.charge = 0.0
        tmp_lib = (output_dir/f"{pdb_file.stem}.tmp.lib")
        tmp_pdb = (output_dir/f"{pdb_file.stem}.tmp.pdb")
        mol2_pmd.save(str(tmp_lib), overwrite=True)
        mol2_pmd.save(str(tmp_pdb), overwrite=True)
        parameterize(
            oligoFile=str(tmp_pdb),
            external_libs=str(tmp_lib), 
            additional_frcmods=str(output_dir/f"{pdb_file.stem}.frcmod"),
            output_dir=output_dir,
            prefix=prefix,
            solvated=False,
            saveparm=True,
            check_atomtypes=False,
            addons=addons
        )
        for tmp_file in [tmp_mol2, tmp_lib, tmp_pdb]:
            if tmp_file.exists():
                os.remove(tmp_file)
    
    @staticmethod
    def calculate_mm_energy(parm, positions: np.ndarray, implicit_solvent=False):
        if not implicit_solvent:
            system = parm.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        else:
            system = parm.createSystem(
                nonbondedMethod=app.CutoffNonPeriodic, constraints=app.HBonds,
                implicitSolvent=app.HCT, useSASA=False) # corresponding to igb=1 in Amber)
        context = mm.Context(system, mm.VerletIntegrator(0.001))
        context.setPositions(positions)
        energy = pmd.openmm.energy_decomposition(parm, context)
        return energy
    
    @staticmethod
    def calculate_mm_energy_sander(parm7: str, rst7: str, workdir="."):
        parm7 = Path(parm7)
        mol_name = parm7.stem
        workdir = Path(workdir)
        cwd = os.getcwd()
        cpptraj_content = f"""parm {parm7} 
        trajin {rst7} 
        esander MOL out {mol_name}.dat gbsa 0 igb 
        """
        with open(workdir/f"{mol_name}.cpptraj.in", "w") as f:
            f.write(cpptraj_content)
        os.chdir(workdir)
        os.system("cpptraj -i cpptraj.in")
        os.chdir(cwd)
        
        
        

