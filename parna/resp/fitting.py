import parmed as pmd
import numpy as np
from rdkit import Chem
from pathlib import Path
import os
import shutil
from collections import OrderedDict
from parna.utils import flatten_list, atomName_to_index, map_atoms, rd_load_file
from parna.logger import getLogger

logger = getLogger(__name__)

PeriodicTable = Chem.GetPeriodicTable()
smartsStringPattern = {
    "NH2": "[#7X3H2]",
    "CH2": "[#6X4H2:1]",
    "CH3": "[#6X4H3]",
    "P": "[#15X4$(*O)]",
    "OE": "[#8X1]",
    "O_P": "[#8X2P2]",
}

def read_chg(fchg):
    charge_info = OrderedDict()
    atom_index = 0
    with open(fchg, "r") as fp:
        for line in fp.readlines():
            info = line.strip().split()
            if len(info) == 5:
                _chagre_info = {}
                _chagre_info["element"] = info[0]
                _chagre_info["corrdinates"] = [float(x) for x in info[1:-1]]
                _chagre_info["charge"] = float(info[-1])
                charge_info[atom_index] = _chagre_info
                atom_index += 1
            else:
                continue
            
    return charge_info


def read_log_chg(log_file):
    charge_info = OrderedDict()
    atom_index = 0
    with open(log_file, "r") as fp:
        content = list(fp.readlines())
    start_line = None
    end_line = None
    for idx, line in enumerate(content):
        if line.strip().startswith("Center       Charge"):
            start_line = idx
        if line.strip().startswith("Sum of charges:"):
            end_line = idx
    assert start_line is not None
    assert end_line is not None
    for line in content[start_line+1:end_line]:
        _chagre_info = {}
        atom_index = int(line[:6].strip())
        _chagre_info["element"] = (line[7:9]).strip()
        _chagre_info["corrdinates"] = None
        _chagre_info["charge"] = float(line.split()[-1])
        charge_info[atom_index-1] = _chagre_info      
    return charge_info


def readCharge(mol2File):
    mol2 = pmd.load_file(mol2File)
    charges = []
    for idx, atom in enumerate(mol2.atoms):
        charges.append([idx+1, atom.charge])
    return charges


def RESPStage1(fchk,  eqConstraints, chgConstraints, a, b,conf_list_file=None, n_proc=1, log_file="multiwfn.log", workdir=None):
    logger.info("Checking existence of Multiwfn...")
    if not shutil.which("Multiwfn"):
        raise FileNotFoundError("Multiwfn is not found in the PATH. Please install Multiwfn and add it to the PATH.")
    multiwfn_script = ["7", "18"]
    # add equivalence constraints
    if eqConstraints is not None:
        multiwfn_script += ["5", "1", str(eqConstraints)]
    # add charge constraints
    if chgConstraints is not None:
        multiwfn_script += ["6", "1", str(chgConstraints)]
    multiwfn_script += ["4", "2", str(a), "0"]
    multiwfn_script += ["4", "1", str(b), "0"]
    # quit multiwfn
    if conf_list_file is not None:
        multiwfn_script += ["-1", str(conf_list_file)]
        multiwfn_script += ["2",]
    else:
        multiwfn_script += ["2", "y"]
    multiwfn_script += ["0", "0", "q"]

    cwd = Path(os.getcwd())
    if workdir is None:
        workdir = cwd
    with open(workdir/"multiwfn.sh", "w") as fp:
        fp.write(f"Multiwfn {fchk} -ispecial 1 -nt {n_proc} << EOF > {log_file}\n")
        fp.write("\n".join(multiwfn_script))
        fp.write("\nEOF")
        
    os.chdir(workdir)
    os.system("bash multiwfn.sh")
    os.chdir(cwd)
   

def File2MOL2(sdf_file, ifiletype, output, atom_type, residue_name, charge=0):
    # by antechamber
    command = [
        "antechamber", "-fi", str(ifiletype), "-i", str(sdf_file),
        "-fo", "mol2", "-o", str(output), "-at", atom_type,
        "-rn", residue_name, "-pf", "y", "-seq", "n", "-nc", str(charge)
    ]
    logger.info(" ".join(command))
    os.system(" ".join(command))


def get_equal_terminal_neighbors(mol, smarts, neighbor_atomic_number):
    pattern_mol = Chem.MolFromSmarts(smarts)
    atoms = mol.GetSubstructMatches(pattern_mol)
    all_pairs = []
    for atom in atoms:
        center = mol.GetAtoms()[atom[0]]
        neighbors = center.GetNeighbors()
        eq_pair = []
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == neighbor_atomic_number \
                    and neighbor.GetDegree() == 1:
                eq_pair.append(neighbor.GetIdx()+1)
        all_pairs.append(eq_pair)
    return all_pairs


def getEquivalenceConstraints(query):
    # find hydrogens in NH2:
    equal_H_NH2 = get_equal_terminal_neighbors(
        query, 
        smartsStringPattern['NH2'], 
        PeriodicTable.GetAtomicNumber("H")
    )
    # find hydrogens in CH2:
    equal_H_CH2 = get_equal_terminal_neighbors(
        query, 
        smartsStringPattern["CH2"],
        PeriodicTable.GetAtomicNumber("H")
    )
    # find hydrogens in CH3:
    equal_H_CH3 = get_equal_terminal_neighbors(
        query, 
        smartsStringPattern["CH3"],
        PeriodicTable.GetAtomicNumber("H")
    )
    return {
        "Stage1": equal_H_NH2,
        "Stage2": equal_H_CH2 + equal_H_CH3
    }

def fit_charges(input_file, wfn_directory, output_dir, residue_name, tightness=0.1, 
                wfn_file_type="molden", wfn_file_prefix="resp_conformer",
                charge_constrained_groups=["OH5", "OH3"], prefix=None):

    constraint_types = {
        "OH5": {
            "atoms": ["O5'", "HO5'"],
            "value": -0.622300 + 0.429500 # -0.1928
        },
        "OH3": {
            "atoms": ["O3'", "HO3'"],
            "value": -0.654100 + 0.437600
        },
    }
    logger.info("Reading input file...")
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    wfn_dir = Path(wfn_directory)
    fchk_files = list(wfn_dir.glob(f"{wfn_file_prefix}*.{wfn_file_type}"))
    conf_list_txt = output_dir/"conformers_list.txt"
    sugar_template_file = Path(__file__).parent.parent/"template/sugar_template.pdb"
    n_conf = len(fchk_files)
    weight_for_each_conf = 1.0/n_conf
    with open(conf_list_txt, "w") as fp:
        for fchk_file in fchk_files:
            fp.write(f"{fchk_file.resolve()} {weight_for_each_conf}\n")
    
    fchk_file = fchk_files[0]
    if str(input_file).endswith(".pdb"):
        query = rd_load_file(str(input_file), removeHs = False, determine_bond_order=False)
    elif str(input_file).endswith(".xyz"):
        query = rd_load_file(str(input_file), determine_bond_order=False)
    elif str(input_file).endswith(".sdf"):
        query = rd_load_file(str(input_file), removeHs = False)
    else:
        raise ValueError("The input file is not supported")
    template = rd_load_file(str(sugar_template_file), removeHs = False)
    
    template_name2idx_map = atomName_to_index(template)
    atom_mapping = map_atoms(template, query)
    
    atom_mapping_dict = {}
    for pair in atom_mapping:
        atom_mapping_dict[pair[0]] = pair[1]
    
    charge_constraints = []
    for constraint_type in charge_constrained_groups:
        atom_names = constraint_types[constraint_type]
        idx_list = [template_name2idx_map[atom_name] for atom_name in atom_names["atoms"]]
        if all([x in atom_mapping_dict.keys() for x in idx_list]):
            atom_pair = [atom_mapping_dict[idx] for idx in idx_list]
            charge_constraints.append([atom_pair, atom_names["value"]])
    
    eq_constraints = getEquivalenceConstraints(query)

    charge_constrains_file = output_dir/f"{Path(input_file).stem}_charge_constraints_stage1.dat"
    equivalence_constraints_1_file = output_dir/f"{Path(input_file).stem}_equivalence_constraints_stage1.dat"
    with open(equivalence_constraints_1_file, "w") as fp:
        for atom_pair in eq_constraints["Stage1"]:
            atom_string = ",".join([str(x) for x in atom_pair])
            fp.write(f"{atom_string}\n")

    with open(charge_constrains_file, "w") as fp:
        for pair in charge_constraints:
            atom_string = ",".join([str(x+1) for x in pair[0]])
            fp.write(f"{atom_string} {pair[1]:.6f}\n")
    
    RESPStage1(
        fchk=fchk_file.resolve(),
        conf_list_file=conf_list_txt.resolve(),
        eqConstraints=equivalence_constraints_1_file.resolve(),
        chgConstraints=charge_constrains_file.resolve(),
        a=0.0005,
        b=tightness,
        n_proc=32,
        log_file=str((output_dir/"multiwfn_stage1.log").resolve()),
        workdir=output_dir.resolve()
    )
    chg_file = output_dir/f"{fchk_file.stem}.stage1.log.chg"
    os.rename(str((output_dir/"multiwfn_stage1.log").resolve()), chg_file)

    
    equivalence_constraints_2_file = output_dir/f"{Path(input_file).stem}_equivalence_constraints_stage2.dat"
    all_equivalence_constraints_stage2 = flatten_list(eq_constraints["Stage2"])

    if ".log." in str(chg_file):
        charge_dict = read_log_chg(chg_file)
    else:
        charge_dict = read_chg(chg_file)
    charge_constraints_stage2 = []
    for i in range(len(charge_dict)):
        # exclude atoms in equivalence constraints
        if not ( i+1 in all_equivalence_constraints_stage2 ):
            charge_constraints_stage2.append([i+1, charge_dict[i]["charge"]])
    
    # all heavy atoms except CH2 and CH3
    charge_constrains_file_2 = output_dir/f"{Path(input_file).stem}_charge_constraints_stage2.dat"
    with open(charge_constrains_file_2, "w") as fp:
        for pair in charge_constraints_stage2:
            fp.write(f"{pair[0]:d} {pair[1]:.6f}\n")
    
    # CH2 and CH3
    equivalence_constraints_2_file = output_dir/f"{Path(input_file).stem}_equivalence_constraints_stage2.dat"
    with open(equivalence_constraints_2_file, "w") as fp:
        for atom_list in eq_constraints["Stage2"]:
            fp.write(f"{','.join([str(a) for a in atom_list])}\n")
    
    RESPStage1(
        fchk=fchk_file.resolve(),
        conf_list_file=conf_list_txt.resolve(),
        eqConstraints=equivalence_constraints_2_file.resolve(),
        chgConstraints=charge_constrains_file_2.resolve(),
        a=0.001,
        b=tightness,
        n_proc=32,
        log_file=str((output_dir/"multiwfn_stage2.log").resolve()),
        workdir=output_dir.resolve()
    )
    chg_file = output_dir/f"{fchk_file.stem}.stage2.log.chg"
    os.rename(str((output_dir/"multiwfn_stage2.log").resolve()), chg_file)
    
    if ".log." in str(chg_file):
        charge_dict = read_log_chg(chg_file)
    else:
        charge_dict = read_chg(chg_file)

    total_charge_mol = sum([x["charge"] for x in charge_dict.values()])
    tmp_pdb = str(output_dir/(input_file.stem+".tmp.pdb"))
    if not str(input_file).endswith(".pdb"):
        Chem.MolToPDBFile(query, tmp_pdb, flavor=2)
        # remove CONECT lines
        with open(tmp_pdb, "r") as fp:
            content = list(fp.readlines())
        with open(tmp_pdb, "w") as fp:
            for line in content:
                if not line.startswith("CONECT"):
                    fp.write(line)
    else:
        shutil.copy(input_file, tmp_pdb)
        
    File2MOL2(
        str(tmp_pdb),
        "pdb",
        (output_dir/f"{Path(input_file).stem}.tmp.mol2").resolve(), 
        "amber",
        str(residue_name),
        charge=int(total_charge_mol)
    )
    
    mol2 = pmd.load_file(str(output_dir/f"{Path(input_file).stem}.tmp.mol2"))
    
    for idx, atom in enumerate(mol2.atoms):
        charge_info = charge_dict[idx]
        assert charge_info["element"] == PeriodicTable.GetElementSymbol(atom.element)
        atom.charge = charge_info["charge"]
        atom.type   = PeriodicTable.GetElementSymbol(atom.element)
    if prefix is not None:
        mol2.save(str(output_dir/f"{prefix}.mol2"), overwrite=True)
    else:
        mol2.save(str(output_dir/f"{Path(input_file).stem}.mol2"), overwrite=True)
    logger.info("File saved to: " + str(output_dir/f"{Path(input_file).stem}.mol2"))
    logger.info(f"Charge fitting finished for {input_file}")
    os.remove(tmp_pdb)
    os.remove(output_dir/f"{Path(input_file).stem}.tmp.mol2")
    



