from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdFMCS
import os
import parmed as pmd
import numpy as np
import yaml
import random
from parna.constant import ATOMIC_NUMBERS

SLURM_HEADER_CPU = """#!/bin/bash
source /etc/profile
module load anaconda/2023a
source activate mdtools
source $HOME/env/multiwfn.env
source $HOME/env/xtb.env
source $HOME/env/orca6.env
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
    # from rdkit.Chem.rdchem import PDBResidueInfo
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
    # atom.HasProp('_TriposAtomName'):
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.HasProp('_TriposAtomName'):
            atomName_to_index[atom.GetProp('_TriposAtomName').strip()] = i
        else:
            atomName_to_index[atom.GetPDBResidueInfo().GetName().strip()] = i
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
        output_file_path = f"{output_folder}/{output_prefix}_{conformer_index:06d}.xyz"

        # Write the conformer data to the individual XYZ file
        with open(output_file_path, 'w') as individual_xyz_file:
            individual_xyz_file.writelines(conformer_data)


def select_from_list(alist, n, method="random"):
    if method == "random":
        return random.sample(alist, n)
    elif method == "first":
        return alist[:n]
    elif method == "last":
        return alist[-n:]
    elif method == "even":
        interval = len(alist) // n
        return alist[::interval]
    else:
        raise ValueError(f"Invalid method: {method}")

def get_suger_picker_angles_from_pseudorotation(phase, intensity, phase_fmt="degree"):
    """
    Get the sugar pucker angles from the pseudorotation phase and intensity.
    Phase in degree, intensity in percentage.
    
    Return:
        - dih1: v1 angle (O4'-C1'-C2'-C3')
        - dih2: v3 angle (C2'-C3'-C4'-O4')
    """
    # convert degree to radian
    if phase_fmt == "degree":
        phase = np.radians(phase)
    
    while phase < 0:
        phase += np.pi*2
    while phase > np.pi*2:
        phase -= np.pi*2
    
    tp = np.tan(phase)
    Zx = np.sqrt(intensity**2 / (1 + tp**2))
    
    if phase > np.pi/2 and phase < np.pi*3/2:
        Zx *= -1.
    Zy = tp * Zx
    dih1 = Zx * np.cos(4*np.pi/5) + Zy * np.sin(4*np.pi/5)
    dih2 = Zx * np.cos(4*np.pi/5) - Zy * np.sin(4*np.pi/5)
    
    if phase_fmt == "degree":
        dih1 = np.degrees(dih1)
        dih2 = np.degrees(dih2)
    
    return dih1, dih2

def save_to_yaml(dict_data, yaml_file):
    with open(yaml_file, "w") as f:
        yaml.dump(dict_data, f)
        

def parse_xyz_file(xyz_file):
    with open(xyz_file, "r") as fp:
        n_atom = int(fp.readline().strip())
        comment = str(fp.readline())
        coords = []
        elements = []
        for i in range(n_atom):
            element, x, y, z = fp.readline().strip().split()
            elements.append(ATOMIC_NUMBERS[element])
            coords.append([float(x),float(y),float(z)])
    
    data = {
        "coord": np.array(coords).reshape(1, n_atom, 3),
        "numbers": np.array(elements).reshape(1, n_atom),
        # "charge": np.array([-1,]).reshape(1,),
        # "mult": np.array([1,])
    }
            
    return data


def dispatch_commands_to_jobs(commands, n_jobs=1, 
                              job_prefix="job", 
                              output_dir=".",
                              work_dir=".",
                              submit=False,
                              submit_options="-s 48"):
    """
    Dispatch a list of commands to multiple jobs.

    Parameters:
    - commands (list): List of commands to be dispatched.
    - n_jobs (int): Number of jobs to dispatch the commands to.
    - job_prefix (str): Prefix of the job script files.
    - output_dir (str): Directory to save the job script files.
    - submit (bool): Whether to submit the jobs to the job
    
    Returns:
    - None
    """
    if len(commands) == 0:
        raise ValueError("The list of commands is empty.")
    if n_jobs < 1:
        raise ValueError("The number of jobs must be greater than 0.")
    
    if len(commands) < n_jobs:
        n_jobs = len(commands)
    
    # Calculate the number of commands per job
    n_commands_each_job = np.ones(n_jobs, dtype=int) * len(commands) // n_jobs
    n_remaining_commands = len(commands) % n_jobs
    n_commands_each_job[:n_remaining_commands] += 1
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True)
    work_dir = Path(work_dir).resolve()
    # Dispatch the commands to jobs
    for i in range(n_jobs):
        # Calculate the start and end indices of the commands for the current job
        if i == 0:
            start_index = 0
        else:
            start_index = sum(n_commands_each_job[:i])
        end_index = start_index + n_commands_each_job[i]

        # Extract the commands for the current job
        job_commands = commands[start_index:end_index]

        # Create a temporary script file for the current job
        script_file_path = output_dir / f"{job_prefix}_{i}.sh"
        with open(script_file_path, 'w') as script_file:
            script_file.write(SLURM_HEADER_CPU)
            script_file.write("\n")
            script_file.write(f"cd {work_dir}\n")
            script_file.write("\n".join(job_commands))
            script_file.write("\n")

    # Dispatch the script file to the job scheduler
    if submit:
        os.chdir(output_dir)
        for i in range(n_jobs):
            script_file_path = f"{job_prefix}_{i}.sh"
            os.system(f"LLsub {script_file_path} {submit_options}")