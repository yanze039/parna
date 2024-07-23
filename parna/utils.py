from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdFMCS
import os
import parmed as pmd
import numpy as np
import yaml
import random

SLURM_HEADER_CPU = """#!/bin/bash
source /etc/profile
module load anaconda/2023a
source activate mdtools
source $HOME/env/multiwfn.env
source $HOME/env/xtb.env
source $HOME/env/orca.env
\n
"""

def rd_load_file(
        infile, 
        charge=None, 
        determine_bond_order=False, 
        atomtype=None, 
        removeHs=False,
        sanitize=True
    ):
    """Load a file into RDKit `Chem.mol` object.
    """
    infile = Path(Path(infile).resolve())
    if infile.suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(infile), removeHs=removeHs, sanitize=sanitize)
        Chem.rdDetermineBonds.DetermineConnectivity(mol)
    elif infile.suffix == ".mol2":
        if atomtype is not None:
            pmd_mol = pmd.load_file(str(infile))
            PeriodicTable = Chem.GetPeriodicTable()
            for atom in pmd_mol.atoms:
                atom.type = PeriodicTable.GetElementSymbol(atom.element)
            tmp_file = str(f"{infile.parent/Path(infile).stem}._tmp.{random.randint(1000, 9999)}.mol2")
            pmd_mol.save(tmp_file, overwrite=True)
            mol = Chem.MolFromMol2File(tmp_file, removeHs=removeHs, sanitize=sanitize)
            os.remove(tmp_file)
        else:
            mol = Chem.MolFromMol2File(str(infile), removeHs=removeHs, sanitize=sanitize)
    elif infile.suffix == ".xyz":
        mol = Chem.MolFromXYZFile(str(infile))
        Chem.rdDetermineBonds.DetermineConnectivity(mol)
    elif infile.suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(infile), removeHs=removeHs, sanitize=sanitize)
        mol = suppl[0]
    else:
        raise ValueError("The input file is not a pdb file")
    if sanitize:
        Chem.SanitizeMol(mol)
    if determine_bond_order:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)

    return mol


def atomName_to_index(mol):
    atomName_to_index = {}
    for i in range(mol.GetNumAtoms()):
        atomName_to_index[mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()] = i
    return atomName_to_index


def map_atoms(template, query, ringMatchesRingOnly=False, \
    bondCompare=rdFMCS.BondCompare.CompareAny, completeRingsOnly=False, atomCompare=rdFMCS.AtomCompare.CompareElements):
    mcs = rdFMCS.FindMCS(
            [template, query], 
            ringMatchesRingOnly=ringMatchesRingOnly,  # PDB doesn't have ring information
         bondCompare=bondCompare,  # PDB doesn't have bond order,
         completeRingsOnly=completeRingsOnly,
         atomCompare=atomCompare
    )
    patt = Chem.MolFromSmarts(mcs.smartsString)
    template_Match = template.GetSubstructMatch(patt)
    query_Match = query.GetSubstructMatch(patt)
    atom_mapping = list(zip(template_Match, query_Match))
    return atom_mapping


def getStringlist(mlist):
    return [str(i) for i in mlist]


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def antechamber(input_file, input_type, output, atom_type, residue_name, charge=0):
    command = [
        "antechamber", "-fi", str(input_type), "-i", str(input_file),
        "-fo", "mol2", "-o", str(output), "-at", atom_type,
        "-rn", residue_name, "-pf", "y", "-seq", "n", "-nc", str(charge)
    ]
    print(" ".join(command))
    os.system(" ".join(command))


def inverse_mapping(mapping):
    if isinstance(mapping, dict):
        return {v: k for k, v in mapping.items()}
    elif isinstance(mapping, list):
        return [(j, i) for i, j in mapping]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    else:
        return v / norm

def read_yaml(YamlFile):
    data = None
    with open(YamlFile, "r") as stream:
        try:
            data = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    if data is None:
        raise ValueError(f"Error in reading {YamlFile}")
    return data

def list2string(alist, delimiter=","):
    return delimiter.join([str(i) for i in alist])

def get_dict_map(data: list):
    return {i: j for i, j in data}

def remove_ter(pdbFile, outFile):
    with open(pdbFile, "r") as f:
        lines = f.readlines()
    with open(outFile, "w") as f:
        for line in lines:
            if not line.startswith("TER"):
                f.write(line)

def merge_list(list1, list2):
    return list(set(list1).union(set(list2)))



def split_xyz_file(input_file_path, output_folder, every_frame=1, output_prefix="conformer"):
    """
    Split a multi-conformer XYZ file into individual XYZ files.

    Parameters:
    - input_file_path (str): Path to the multi-conformer XYZ file.
    - output_folder (str): Folder where individual XYZ files will be saved.

    Returns:
    - None
    """
    # Read the content of the multi-conformer XYZ file
    with open(input_file_path, 'r') as multi_xyz_file:
        lines = multi_xyz_file.readlines()

    # Find the number of conformers
    block_length = int(lines[0].strip()) + 2
    num_conformers = len(lines) // block_length
    
    num_conformers = num_conformers // every_frame
    # Iterate over conformers and create individual XYZ files
    for _conformer_index in range(num_conformers):
        # Extract the data for the current conformer
        conformer_index = _conformer_index * every_frame
        start_index = (conformer_index) * block_length
        end_index = start_index + block_length
        conformer_data = lines[start_index:end_index]

        # Generate the output file path for the current conformer
        output_file_path = f"{output_folder}/{output_prefix}_{conformer_index}.xyz"

        # Write the conformer data to the individual XYZ file
        with open(output_file_path, 'w') as individual_xyz_file:
            individual_xyz_file.writelines(conformer_data)


