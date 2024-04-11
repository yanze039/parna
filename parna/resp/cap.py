import parmed as pmd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import Counter
from pathlib import Path
import os
from collections import OrderedDict


PeriodicTable = Chem.GetPeriodicTable()
smartsStringPattern = {
    "NH2": "[#7X3H2]",
    "CH2": "[#6X4H2:1]",
    "CH3": "[#6X4H3]",
    "P": "[#15D4]",
    "OE": "[#8X1]",
    "O_P": "[#8X2P2]",
    "OCH3": "O[#6X4H3]",
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


def readCharge(mol2File):
    mol2 = pmd.load_file(mol2File)
    charges = []
    for idx, atom in enumerate(mol2.atoms):
        charges.append([idx+1, atom.charge])
    return charges


def RESPStage1(fchk, eqConstraints, chgConstraints, a, b, n_proc=1, log_file="multiwfn.log", workdir=None):
    multiwfn_script = ["7", "18"]
    # add equivalence constraints
    cwd = os.getcwd()
    if workdir is not None:
        os.chdir(workdir)
    
    if eqConstraints is not None:
        multiwfn_script += ["5", "1", str(eqConstraints)]
    # add charge constraints
    if chgConstraints is not None:
        multiwfn_script += ["6", "1", str(chgConstraints)]
    multiwfn_script += ["4", "2", str(a), "0"]
    multiwfn_script += ["4", "1", str(b), "0"]
    # quit multiwfn
    multiwfn_script += ["2", "y", "0", "0", "q"]
    with open("multiwfn.sh", "w") as fp:
        fp.write(f"Multiwfn {fchk} -ispecial 1 -nt {n_proc} << EOF > {log_file}\n")
        fp.write("\n".join(multiwfn_script))
        fp.write("\nEOF")
    os.system("bash multiwfn.sh")
    os.chdir(cwd)


def flatten_list(l):
    return [item for sublist in l for item in sublist]

def File2MOL2(sdf_file, ifiletype, output, atom_type, residue_name):
    # by antechamber
    command = [
        "antechamber", "-fi", str(ifiletype), "-i", str(sdf_file),
        "-fo", "mol2", "-o", str(output), "-at", atom_type,
        "-rn", residue_name, "-pf", "y", "-seq", "n"
    ]
    print(" ".join(command))
    os.system(" ".join(command))

        

def get_name_idx_mapping(mol):
    name_idx_mapping = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_name = (atom.GetMonomerInfo().GetName()).strip()
        name_idx_mapping[atom_name] = idx
    return name_idx_mapping

def map_atoms(template, query):
    mcs = rdFMCS.FindMCS(
            [template, query], 
            ringMatchesRingOnly=False,
         bondCompare=rdFMCS.BondCompare.CompareAny  # PDB doesn't have bond order
    )
    patt = Chem.MolFromSmarts(mcs.smartsString)
    template_Match = template.GetSubstructMatch(patt)
    query_Match = query.GetSubstructMatch(patt)
    atom_mapping = list(zip(template_Match, query_Match))
    return atom_mapping

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
    # find oxygens in PO4:
    equal_O_PO4 = get_equal_terminal_neighbors(
        query, 
        smartsStringPattern["P"], 
        PeriodicTable.GetAtomicNumber("O")
    )
    return {
        "Stage1": equal_H_NH2 + equal_O_PO4,
        "Stage2": equal_H_CH2 + equal_H_CH3
    }


def getMethylChargeConstraints(query, target_charge):
    # read sdf file
    # find CH3 pattern
    pattern_mol = Chem.MolFromSmarts(smartsStringPattern["OCH3"])
    carbon_atoms = query.GetSubstructMatches(pattern_mol)
    charge_constraints = []
    for c in carbon_atoms:
        # get hydrogens
        atom = query.GetAtoms()[c[1]]
        neighbors = atom.GetNeighbors()
        neighbor_hydrogens = []
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == PeriodicTable.GetAtomicNumber("H"):
                neighbor_hydrogens.append(neighbor.GetIdx())
        # set charge constraints
        charge_constraints.append([[c[1]+1]+[h+1 for h in neighbor_hydrogens], target_charge])
    return charge_constraints

def fit_charges_cap(input_file, wfn_file, output_dir, residue_name, tightness=0.1):
    # methyl + 0.4295 -0.6223 = 0
    # methyl = -0.4295 + 0.6223 = 0.1928
    methyl_charge = 0.1928
    pdbfile = input_file
    fchk_file = Path(wfn_file)
    output_dir = Path(output_dir)

    tightness = tightness
    query = Chem.MolFromPDBFile(pdbfile, removeHs=False)
    # rdDetermineBonds.DetermineConnectivity(query)
    charge_constraints1 = getMethylChargeConstraints(query, methyl_charge)
    
    eq_constraints = getEquivalenceConstraints(query)

    charge_constrains_file = output_dir/f"{Path(pdbfile).stem}_charge_constraints_stage1.dat"
    equivalence_constraints_1_file = output_dir/f"{Path(pdbfile).stem}_equivalence_constraints_stage1.dat"
    with open(equivalence_constraints_1_file, "w") as fp:
        for atom_pair in eq_constraints["Stage1"]:
            atom_string = ",".join([str(x) for x in atom_pair])
            fp.write(f"{atom_string}\n")

    with open(charge_constrains_file, "w") as fp:
        for pair in charge_constraints1:
            atom_string = ",".join([str(x) for x in pair[0]])
            fp.write(f"{atom_string} {pair[1]:.6f}\n")
    
    # if not os.path.exists(f"{fchk_file.stem}.stage1.chg"):
    RESPStage1(
        fchk=fchk_file.resolve(),
        eqConstraints=equivalence_constraints_1_file.resolve(),
        chgConstraints=charge_constrains_file.resolve(),
        a=0.0005,
        b=tightness,
        n_proc=32,
        log_file=str((output_dir/"multiwfn_stage1.log").resolve()),
        workdir=output_dir
    )
    os.rename(output_dir/f"{fchk_file.stem.split('.')[0]}.chg", output_dir/f"{fchk_file.stem}.stage1.chg")

    
    equivalence_constraints_2_file = output_dir/f"{Path(pdbfile).stem}_equivalence_constraints_stage2.dat"
    all_equivalence_constraints_stage2 = flatten_list(eq_constraints["Stage2"])

    chg_file = output_dir/f"{fchk_file.stem}.stage1.chg"
    charge_dict = read_chg(chg_file)
    charge_constraints_stage2 = []
    for i in range(len(charge_dict)):
        # exclude atoms in equivalence constraints
        if not ( i+1 in all_equivalence_constraints_stage2 ):
            charge_constraints_stage2.append([i+1, charge_dict[i]["charge"]])
    
    # all heavy atoms except CH2 and CH3
    charge_constrains_file_2 = output_dir/f"{Path(pdbfile).stem}_charge_constraints_stage2.dat"
    with open(charge_constrains_file_2, "w") as fp:
        for pair in charge_constraints_stage2:
            fp.write(f"{pair[0]:d} {pair[1]:.6f}\n")
        for pair in charge_constraints1:
            atom_string = ",".join([str(x) for x in pair[0]])
            fp.write(f"{atom_string} {pair[1]:.6f}\n")
        
    
    # CH2 and CH3
    equivalence_constraints_2_file = output_dir/f"{Path(pdbfile).stem}_equivalence_constraints_stage2.dat"
    with open(equivalence_constraints_2_file, "w") as fp:
        for atom_list in eq_constraints["Stage2"]:
            fp.write(f"{','.join([str(a) for a in atom_list])}\n")
    RESPStage1(
        fchk=fchk_file.resolve(),
        eqConstraints=equivalence_constraints_2_file.resolve(),
        chgConstraints=charge_constrains_file_2.resolve(),
        a=0.001,
        b=tightness,
        n_proc=32,
        log_file=str((output_dir/"multiwfn_stage2.log").resolve()),
        workdir=output_dir
    )
    os.rename(output_dir/f"{fchk_file.stem.split('.')[0]}.chg", output_dir/f"{fchk_file.stem}.stage2.chg")
    
    
    File2MOL2(
        str(pdbfile),
        "pdb",
        output_dir/f"{Path(pdbfile).stem}.tmp.mol2", 
        "gaff2", 
        "M7G"
    )
    mol2 = pmd.load_file(str(output_dir/f"{Path(pdbfile).stem}.tmp.mol2"))
    chg_file = output_dir/f"{fchk_file.stem}.stage2.chg"
    charge_dict = read_chg(chg_file)
    for idx, atom in enumerate(mol2.atoms):
        charge_info = charge_dict[idx]
        assert charge_info["element"] == PeriodicTable.GetElementSymbol(atom.element)
        atom.charge = charge_info["charge"]
    mol2.save(str(output_dir/f"{Path(pdbfile).stem}.{tightness:.3f}.mol2"), overwrite=True)
    os.remove(output_dir/f"{Path(pdbfile).stem}.tmp.mol2")



    