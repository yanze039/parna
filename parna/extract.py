import os
from rdkit import Chem
from rdkit.Chem import rdFMCS
import parmed as pmd
from pathlib import Path
from parna.utils import read_yaml
from parna.molops import construct_local_frame
import yaml
import numpy as np
import shutil


DATA = Path(__file__).parent / "data"
TEMPLATE = Path(__file__).parent / "template"
FRAG = Path(__file__).parent/"fragments"
FRAME = Path(__file__).parent/"local_frame"

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


def atomName_to_index(mol):
    atomName_to_index = {}
    for i in range(mol.GetNumAtoms()):
        atomName_to_index[mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()] = i
    return atomName_to_index


def extract_residue(residue_template, residue_name, output_dir, keep_hydrogen=False, with_phosphate=True, canonical_residue=True):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    if not keep_hydrogen:
        residue_template.strip("@H*")
    residue_template.write_pdb(str(output_dir/f"{residue_name}.pdb"), use_hetatoms=False)
    if not canonical_residue:
        return None

    rd_mol = Chem.MolFromPDBFile(str(output_dir/f"{residue_name}.pdb"), removeHs=False, sanitize=False)
    if with_phosphate:
        sugar_template = Chem.MolFromPDBFile(str(TEMPLATE/"sugar_template_with_PO3.pdb"), removeHs=True, sanitize=False)
    else:
        sugar_template = Chem.MolFromPDBFile(str(TEMPLATE/"sugar_template.pdb"), removeHs=True, sanitize=False)
    atom_maps = map_atoms(sugar_template, rd_mol)
    
    mapping_dict = {}
    for atom in atom_maps:
        mapping_dict[atom[0]] = atom[1]
    name2idx = atomName_to_index(sugar_template)
    sugaratom = mapping_dict[name2idx["C1'"]]
    if name2idx["N9"] in mapping_dict.keys():
        center = mapping_dict[name2idx["N9"]]
    else:
        C1_neighbors = rd_mol.GetAtomWithIdx(sugaratom).GetNeighbors()
        C1_neighbors_idx = [atom.GetIdx() for atom in C1_neighbors]
        base_atom_1_candidate = [idx for idx in C1_neighbors_idx if idx not in mapping_dict.values()]
        if len(base_atom_1_candidate) != 1:
            raise ValueError("The number of base atom 1 is not 1")
        center = base_atom_1_candidate[0]
    
    # save backbone block
    backbone_block = Chem.RWMol()
    rd_mol_conf = rd_mol.GetConformer()
    backbone_conf = Chem.Conformer(backbone_block.GetNumAtoms())
    attach_atom = None
    for idx, atom in enumerate(mapping_dict.values()):
        if atom == center:
            continue
        if atom == sugaratom:
            attach_atom = idx
        backbone_block.AddAtom(rd_mol.GetAtomWithIdx(atom))
        backbone_conf.SetAtomPosition(idx, rd_mol_conf.GetAtomPosition(atom))
    backbone_block.AddConformer(backbone_conf)
    Chem.MolToPDBFile(backbone_block, str(output_dir/f"{residue_name}_backbone.pdb"))

    center_neighbors = rd_mol.GetAtomWithIdx(center).GetNeighbors()
    center_neighbors = [atom.GetIdx() for atom in center_neighbors if atom.GetIdx() != sugaratom]
    if len(center_neighbors) != 2:
        raise ValueError("The number of center neighbors is not 2")
    cn_with_degrees = []
    for cn in center_neighbors:
        cn_with_degrees.append((cn, len(rd_mol.GetAtomWithIdx(cn).GetNeighbors())))
    cn_with_degrees.sort(key=lambda x: x[1], reverse=True)
    ringatom1 = cn_with_degrees[0][0]
    ringatom2 = cn_with_degrees[1][0]
    
    if attach_atom is None:
        raise ValueError("attach_atom is None")
    return {
        "sugaratom": sugaratom,
        "center": center,
        "ringatom1": ringatom1,
        "ringatom2": ringatom2,
        "attach_atom": attach_atom
    }


def extract_backbone(input_file, output_dir, 
                     start=0, end=-1, 
                     noncanonical_residues=[], 
                     residues_without_phosphate=[], 
                     keep_hydrogen=False):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    pmd_mol = pmd.load_file(str(input_file))
    n_residues = len(pmd_mol.residues)
    attach_atoms = {}
    local_frame_backbone = {}

    if end < 0:
        end = (n_residues + end) % n_residues
    for i in range(n_residues):
        if i < start:
            continue
        if i > end:
            break
        print(f"Extracting residue {i} ...")
        residue = pmd_mol.residues[i]
        residue_template = pmd.modeller.residue.ResidueTemplate().from_residue(residue).to_structure()
        residue_name=residue.name+"_"+str(residue.number)
        canonical_residue=(residue.number not in noncanonical_residues)
        frame_info = extract_residue(residue_template, 
                        residue_name=residue_name,
                        output_dir=output_dir, 
                        keep_hydrogen=keep_hydrogen, 
                        with_phosphate=(residue.number not in residues_without_phosphate),
                        canonical_residue=canonical_residue
        )
        if canonical_residue:
            shutil.copy(
                str(output_dir/f"{residue_name}_backbone.pdb"),
                str(output_dir/f"N{i}_backbone.pdb")
            )
        if frame_info is None:
            continue
        
        attach_atoms[f"N{i}"] = frame_info["attach_atom"]
        attach_atoms[residue_name] = frame_info["attach_atom"]
        local_frame_backbone[residue_name] = {}
        local_frame_backbone[residue_name]["sugaratom"] = {"index": frame_info["sugaratom"]}
        local_frame_backbone[residue_name]["center"] = {"index": frame_info["center"]}
        local_frame_backbone[residue_name]["ringatom1"] = {"index": frame_info["ringatom1"]}
        local_frame_backbone[residue_name]["ringatom2"] = {"index": frame_info["ringatom2"]}
        
        # get backbone local frame
        bb_frame = Chem.MolFromPDBFile(str(output_dir/f"{residue_name}.pdb"))
        vector1, norm_vec = construct_local_frame(
            bb_frame, frame_info["center"], frame_info["ringatom1"], 
            frame_info["ringatom2"], frame_info["sugaratom"]
        )
        triangle = (np.stack([vector1, norm_vec], axis=0))
        np.save(output_dir/f"{residue_name}_frame.npy", triangle)
        np.save(output_dir/f"N{i}_frame.npy", triangle)
        # get attach point
        attech_point = bb_frame.GetConformer().GetAtomPosition(frame_info["center"])
        np.save(output_dir/f"{residue_name}_attach_point.npy", np.array(attech_point))
        np.save(output_dir/f"N{i}_attach_point.npy", np.array(attech_point))
    
    print("Saving attach_atoms.yaml and local_frame_backbone.yaml ...")
    print("File path:", output_dir.resolve())
    with open(output_dir/"attach_atoms.yaml", "w") as f:
        yaml.dump(attach_atoms, f)
    with open(output_dir/"local_frame_backbone.yaml", "w") as f:
        yaml.dump(local_frame_backbone, f)


def extract_residue_from_pdb(input_file, output_file, residue_number):
    pmd_mol = pmd.load_file(str(input_file))
    residue_template = pmd.modeller.residue.ResidueTemplate().from_residue(pmd_mol.residues[residue_number-1]).to_structure()
    residue_template.write_pdb(str(output_file), use_hetatoms=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input file")
    parser.add_argument("-o", "--output_dir", type=str, help="input file")
    parser.add_argument("-nc", "--noncanonical_residues", default=None, type=int, nargs="+", help="noncanonical residues")
    args = parser.parse_args()
    if args.noncanonical_residues is None:
        args.noncanonical_residues = []
    input_file = args.input
    extract_backbone(
        input_file, 
        output_dir=args.output_dir, 
        noncanonical_residues=args.noncanonical_residues, 
        residues_without_phosphate=[], 
        keep_hydrogen=False
    )
