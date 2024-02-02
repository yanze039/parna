from rdkit import Chem
from rdkit.Chem import rdFMCS
import numpy as np
import psi4
from typing import List
import os
from pathlib import Path
import argparse 
from parna.logger import getLogger

logger = getLogger()


def mk_psi4_geometry(atom_names: List[str], coords: np.ndarray, charge: int, mult: int):

    ret = f"{charge} {mult}\n"
    for name, coord in zip(atom_names, coords):
        ret += f"{name} {coord[0]:>10.6f} {coord[1]:>10.6f} {coord[2]:>10.6f}\n"
    mol = psi4.geometry(ret)
    return mol


def calculate_energy(input_file,
                     output_dir, 
                     charge=0, 
                     memory="160 GB", 
                     n_threads=48, 
                     method_basis="HF/6-31G*"):
    
    logger.info("calculating HF/6-31G* energy for " + str(input_file))
    output_dir = Path(output_dir)
    inFile = Path(input_file)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    psi4log = output_dir/f"{inFile.stem}.psi4.HF.log"
    psi4tmp_dir = output_dir/"tmp"
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)
    psi4.core.set_output_file(str(psi4log), False)
    psi4_io = psi4.core.IOManager.shared_object()
    if not os.path.exists(psi4tmp_dir):
        os.makedirs(psi4tmp_dir, exist_ok=True)
    psi4_io.set_default_path(str(psi4tmp_dir.resolve()))

    if inFile.suffix == ".pdb":
        mymol = Chem.MolFromPDBFile(str(inFile), removeHs = False)
    elif inFile.suffix == ".xyz":
        mymol = Chem.MolFromXYZFile(str(inFile))
    else:
        raise ValueError("The input file is not supported")
    coords = mymol.GetConformer().GetPositions()
    atom_symbols = [atom.GetSymbol() for atom in mymol.GetAtoms()]
  
    # get heavy atoms, set frozen cartesian, atom id starts from 1
    psi4_mol = mk_psi4_geometry(
        atom_symbols, coords, charge, 1
    )
    logger.info("Running psi4...")

    energy, wfn = psi4.energy(method_basis, molecule=psi4_mol, return_wfn=True)
    psi4.driver.fchk(wfn, str(output_dir/f"{inFile.stem}.psi4.HF.fchk"))
    return energy

