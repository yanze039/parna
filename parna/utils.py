from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdFMCS
import os
import parmed as pmd
import numpy as np
import yaml


def rd_load_file(
        infile, 
        charge=None, 
        determine_bond_order=False, 
        atomtype=None, 
        removeHs=False,
        sanitize=True
    ):
    infile = Path(infile)
    if infile.suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(infile), removeHs=removeHs, sanitize=sanitize)
    elif infile.suffix == ".mol2":
        if atomtype is not None:
            pmd_mol = pmd.load_file(str(infile))
            PeriodicTable = Chem.GetPeriodicTable()
            for atom in pmd_mol.atoms:
                atom.type = PeriodicTable.GetElementSymbol(atom.element)
            tmp_file = str(f"{infile.parent/Path(infile).stem}._tmp.mol2")
            pmd_mol.save(tmp_file, overwrite=True)
            mol = Chem.MolFromMol2File(tmp_file, removeHs=removeHs, sanitize=sanitize)
            os.remove(tmp_file)
        else:
            mol = Chem.MolFromMol2File(str(infile), removeHs=removeHs, sanitize=sanitize)
    elif infile.suffix == ".xyz":
        mol = Chem.MolFromXYZFile(str(infile))
    else:
        raise ValueError("The input file is not a pdb file")
    if determine_bond_order:
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
    return mol


def atomName_to_index(mol):
    atomName_to_index = {}
    for i in range(mol.GetNumAtoms()):
        atomName_to_index[mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()] = i
    return atomName_to_index


def map_atoms(template, query, ringMatchesRingOnly=False, bondCompare=rdFMCS.BondCompare.CompareAny):
    mcs = rdFMCS.FindMCS(
            [template, query], 
            ringMatchesRingOnly=ringMatchesRingOnly,  # PDB doesn't have ring information
         bondCompare=bondCompare  # PDB doesn't have bond order
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

def remove_ter(pdbFile, outFile):
    with open(pdbFile, "r") as f:
        lines = f.readlines()
    with open(outFile, "w") as f:
        for line in lines:
            if not line.startswith("TER"):
                f.write(line)
