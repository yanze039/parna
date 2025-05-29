from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdFMCS
import os
import parmed as pmd
import numpy as np
import yaml
import random
import openfe
from parna.constant import ATOMIC_NUMBERS
from parna.atom_mapper import FuzzyElementCompareAtoms, CompareChiralElements

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
                #BUG: a known bug. Parmed does not recognize Br and Cl. The `atom.element` 
                # will be "B" and "C" instead of "Br" and "Cl".
                if atom.type.strip().lower() in ["br", "cl"]:
                    atom.type = atom.type
                else:
                    atom.type = PeriodicTable.GetElementSymbol(atom.element)
            tmp_mol2_file = str(f"{infile.parent/Path(infile).stem}._tmp.{random.randint(1000, 9999)}.mol2")
            tmp_pdb_file  = str(f"{infile.parent/Path(infile).stem}._tmp.{random.randint(1000, 9999)}.pdb")
            pmd_mol.save(tmp_mol2_file, overwrite=True)
            pmd_mol.save(tmp_pdb_file, overwrite=True)
            mol = Chem.MolFromMol2File(tmp_mol2_file, removeHs=removeHs, sanitize=sanitize)
            _mol = Chem.MolFromPDBFile(tmp_pdb_file, removeHs=removeHs, sanitize=sanitize)
            for i in range(mol.GetNumAtoms()):
                # set residue info
                mol.GetAtomWithIdx(i).SetPDBResidueInfo(_mol.GetAtomWithIdx(i).GetPDBResidueInfo())
            os.remove(tmp_mol2_file)
            os.remove(tmp_pdb_file)
            del _mol
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


def get_mapping_from_pattern(pattern, mol1, mol2, useChirality=True):
    template_Match = mol1.GetSubstructMatch(pattern, useChirality=useChirality)
    query_Match = mol2.GetSubstructMatch(pattern, useChirality=useChirality)
    atom_mapping = list(zip(template_Match, query_Match))
    return atom_mapping


class EnforceMappingAcceptance(rdFMCS.MCSAcceptance):
    
    def register_atom_mapping(self, custom_map):
        self.custom_map = custom_map
    
    def __call__(self, mol1, mol2, atom_idx_match, bond_idx_match, params):
        atom_idx_match_dict = {x[0]: x[1] for x in atom_idx_match}
        for mol1_atom_idx, mol2_atom_idx in self.custom_map:
            # if mol1.GetAtomWithIdx(mol1_atom_idx).GetAtomicNum() == 0:
            #     return True 
            # if mol2.GetAtomWithIdx(mol2_atom_idx).GetAtomicNum() == 0:
            #     return True
            if mol1_atom_idx in atom_idx_match_dict and atom_idx_match_dict[mol1_atom_idx] == mol2_atom_idx:
                continue
            else:
                return False
        return True

def constrained_map_atoms(
        template, query, constrained_mapping, 
        ringMatchesRingOnly=False, 
        bondCompare=rdFMCS.BondCompare.CompareAny, 
        completeRingsOnly=False, 
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        seed=None
    ):
    
    params = rdFMCS.MCSParameters()
    params.BondTyper = bondCompare
    fuzzy_compare = FuzzyElementCompareAtoms(
        comparison=atomCompare,
        custom_map=constrained_mapping,
        n_atoms_mol1=template.GetNumAtoms(),
        n_atoms_mol2=query.GetNumAtoms()
    )
    params.AtomTyper = fuzzy_compare
    
    params.CompleteRingsOnly = completeRingsOnly
    params.RingMatchesRingOnly = ringMatchesRingOnly
    # params.Verbose = True
    if seed is not None:
        params.InitialSeed = seed
    # print("params:", params.CompleteRingsOnly, params.CompleteRingsOnly)
    compare = [template, query]
    res: rdFMCS.MCSResult = rdFMCS.FindMCS(compare, params)
    atom_maps = get_mapping_from_pattern(Chem.MolFromSmarts(res.smartsString),
                                        compare[0], compare[1])
    return atom_maps
    
    
def map_atoms(template, query, ringMatchesRingOnly=False, \
        bondCompare=rdFMCS.BondCompare.CompareAny, 
        completeRingsOnly=False, 
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        fuzzy_matching=False,
        matchChiralTag=False,
    ):
    
    params = rdFMCS.MCSParameters()
    params.BondTyper = bondCompare
    
    if matchChiralTag:
        Chem.AssignStereochemistryFrom3D(template)
        Chem.AssignStereochemistryFrom3D(query)
        # Chem.AssignStereochemistry(template, force=True, cleanIt=True)
        # Chem.AssignStereochemistry(query, force=True, cleanIt=True)
        atomCompare = CompareChiralElements(
            Chem.FindMolChiralCenters(template, includeUnassigned=True),
            Chem.FindMolChiralCenters(query, includeUnassigned=True)
        )
    params.AtomTyper = atomCompare
    params.CompleteRingsOnly = completeRingsOnly
    params.RingMatchesRingOnly = ringMatchesRingOnly
    params.matchChiralTag = matchChiralTag
    params.MatchChiralTag = matchChiralTag
    
    mcs = rdFMCS.FindMCS(
        [template, query], 
        params
    )
    patt = Chem.MolFromSmarts(mcs.smartsString)
    atom_mapping = get_mapping_from_pattern(
        patt, template, query, useChirality=matchChiralTag
    )
    
    if fuzzy_matching:
        atom_mapping = constrained_map_atoms(
            template, query, atom_mapping, 
            ringMatchesRingOnly=ringMatchesRingOnly, 
            bondCompare=rdFMCS.BondCompare.CompareAny, 
            completeRingsOnly=completeRingsOnly, 
            atomCompare=rdFMCS.AtomCompare.CompareAny,
            seed=mcs.smartsString
        )
    return atom_mapping


def map_atoms_openfe(template, query, element_change=True):
    
    tmpl_comp = openfe.SmallMoleculeComponent.from_rdkit(template)
    query_comp = openfe.SmallMoleculeComponent.from_rdkit(query)

    mapper = openfe.LomapAtomMapper(max3d=1000.0, element_change=element_change)
    mapping = next(mapper.suggest_mappings(tmpl_comp, query_comp))
    mapping_candidate = (mapping).componentA_to_componentB
    return [(i, j) for i, j in mapping_candidate.items()]
    

def getStringlist(mlist):
    return [str(i) for i in mlist]


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def antechamber(input_file, input_type, output, atom_type, residue_name, charge=0):
    command = [
        "antechamber", "-fi", str(input_type), "-i", str(input_file),
        "-fo", "mol2", "-o", str(output), "-at", atom_type,
        "-rn", residue_name, "-pf", "y", "-seq", "n", "-nc", str(charge),
        "-dr", "no"
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