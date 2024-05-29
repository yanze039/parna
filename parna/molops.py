import numpy as np
import rdkit
from rdkit import Chem
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
import shutil


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

        
    