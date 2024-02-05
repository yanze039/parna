import parmed as pmd
import rdkit.Chem as Chem
from pathlib import Path
import os
import numpy as np

from parna.utils import antechamber, atomName_to_index, map_atoms, rd_load_file, read_yaml
from parna.molops import flexible_align
from parna.molops import ortho_frame, rotate_conformer, translate_conformer

from parna.logger import getLogger

logger = getLogger(__name__)

template_dir = Path(__file__).parent/"template"
FRAG = Path(__file__).parent/"fragments"
DATA = Path(__file__).parent/"data"
FRAME = Path(__file__).parent/"local_frame"


O3_charge = -0.5246
O5_charge = -0.4989
P_charge = 1.1662
OP1_charge = -0.7760
OP2_charge = -0.7760


def build_backbone_block(base_name: str, backbone_position: int):

    baseFile = FRAG / f"Base_{base_name}_hvy.pdb"
    backboneFile = FRAG / f"N{backbone_position}_backbone.pdb"
    # read PDB file
    base = Chem.MolFromPDBFile(str(baseFile), sanitize=False)
    backbone = Chem.MolFromPDBFile(str(backboneFile), sanitize=False)
    # read parameters
    attach_atoms = read_yaml(DATA/"attach_atoms.yaml")
    if attach_atoms is None:
        raise ValueError("attach_atoms.yaml is not found")
    bb_anchor_atom_pos = np.load(FRAME/f"N{backbone_position:d}_attach_point.npy")
    base_local_frame = np.load(FRAME/f"{base_name}_frame.npy")
    backbone_local_frame = np.load(FRAME/f"N{backbone_position:d}_frame.npy")
    # get new ortho basis from local frame, this is because N is not perfectly sp2 hybridized (planar).
    base_basis = ortho_frame(base_local_frame[0], base_local_frame[1])
    backbone_basis = ortho_frame(backbone_local_frame[0], backbone_local_frame[1])
    # now rotate the base to align to backbone, eqivalent to two changes of basis.
    # in theory, we rotate M1 back and rotate to M2, so backbone_basis @ np.linalg.inv(base_basis) ,
    # but remember coordinates are row vectors, therefore we transpose the vector above to get:
    # np.linalg.inv(backbone_basis) @ (base_basis)
    final_transform = np.matmul( np.linalg.inv(backbone_basis), (base_basis) )
    rotate_conformer(base.GetConformer(), final_transform)
    base_anchor_atom_pos = np.array(base.GetConformer().GetAtomPosition(attach_atoms[base_name]))
    # translate molecules to backbone
    tranlation_vec = bb_anchor_atom_pos - base_anchor_atom_pos
    translate_conformer(base.GetConformer(), tranlation_vec)
    # combine the two fragments, connecting bonds
    nucleoside = Chem.CombineMols(backbone,base)
    edmol = Chem.EditableMol(nucleoside)
    edmol.AddBond(
        attach_atoms[f"N{backbone_position:d}"],
        attach_atoms[base_name]+backbone.GetNumAtoms(),
        order=Chem.rdchem.BondType.SINGLE
    )
    finalmol = edmol.GetMol()
    return finalmol


def build_backbone(sequence, chain_id = "A"):
    cap_file = FRAG / "m7gppp.pdb"
    if sequence[0] == "cap":
        print("The oligo uses m7G-capped structure.")
        template = Chem.MolFromPDBFile(str(template_dir/"m7gppp.pdb"), removeHs=False)
        atom_maps = map_atoms(
            template,
            Chem.MolFromPDBFile(str(cap_file), removeHs=False)
        )
        myStructure = pmd.load_file(str(cap_file))
        for template_i, block_j in atom_maps:
            myStructure.atoms[block_j].name = template.GetAtoms()[template_i].GetPDBResidueInfo().GetName()
        sequence = sequence[1:]
        myStructure.residues[0].chain = chain_id
        myStructure.strip("@AT")
    else:
        myStructure = pmd.Structure()
    for pos, basename in enumerate(sequence, start=1):
        print(f"Building residue {pos} with base {basename}")
        _basename = basename[0]
        rd_block = build_backbone_block(_basename, pos)
        pmd_block = pmd.rdkit.load_rdkit( rd_block )
        template = Chem.MolFromPDBFile(str(template_dir/f"{basename}.pdb"), removeHs=False)
        atom_maps = map_atoms(template, rd_block)
        for template_i, block_j in atom_maps:
            atom = pmd_block.atoms[block_j]
            atom.name = template.GetAtoms()[template_i].GetPDBResidueInfo().GetName()
            myStructure.add_atom(atom, 
                                 resname=basename,
                                 resnum=pos,
                                 chain=chain_id
                                 )
    return myStructure


def build_residue_pdb_block(input_file, residue_name, template_residue, atomtype=None):

    rd_mol = rd_load_file(input_file, removeHs=False, atomtype=atomtype)
    if not "." in str(template_residue):
        rd_tmpl_block = build_backbone_block(template_residue, 1)
    else:
        rd_tmpl_block = Chem.MolFromPDBFile(str(template_residue), removeHs=False, sanitize=False)
    atom_mapping = map_atoms(rd_tmpl_block, rd_mol)
    
    inverse_atom_mapping = [(j, i) for i, j in atom_mapping]
    rd_mol = flexible_align(
        rd_mol,
        rd_tmpl_block,
        atom_mapping=inverse_atom_mapping,
        force_constant=100.0,
    )
    # Find BUG: the atom names are not preserved from mol2. we use parmed to fix this.
    # pdbblock = Chem.MolToPDBBlock(rd_mol)
    pmd_mol = pmd.rdkit.load_rdkit(rd_mol)
    old_mol = pmd.load_file(input_file)
    for idx, atom in enumerate(pmd_mol.atoms):
        atom.name = old_mol.atoms[idx].name
    pmd_mol.write_pdb("_tmp.pdb")
    with open("_tmp.pdb", "r") as f:
        pdbblock = f.read()
    atom_list = []
    for line in pdbblock.split("\n"):
        if line.startswith("ATOM"):
            line = line[:17] + f"{residue_name:>3}" + line[20:]
            atom_list.append(line)
        elif line.startswith("HETATM"):
            line = "ATOM  " + line[6:17] + f"{residue_name:>3}" + line[20:]
            atom_list.append(line)
    os.remove("_tmp.pdb")
    return "\n".join(atom_list)


def build_residue_without_phosphate(input_file, atom_name_mapping, residue_name, charge):
    antechamber(
        input_file=input_file, 
        input_type=input_file.suffix[1:], 
        output=input_file.parent/f"{residue_name}._amber.mol2", 
        atom_type="amber", 
        residue_name=residue_name,
        charge=charge
    )
    pmd_mol = pmd.load_file(str(input_file.parent/f"{residue_name}._amber.mol2"))
    pmd_mol.fix_charges()
    pmd_mol.atoms[atom_name_mapping["O3'"]].charge = O3_charge
    pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OS"
    pmd_mol.head = pmd_mol.atoms[atom_name_mapping["C5'"]]
    pmd_mol.tail = pmd_mol.atoms[atom_name_mapping["O3'"]]
    atom_delete = [
        pmd_mol.atoms[atom_name_mapping["HO3'"]],
        pmd_mol.atoms[atom_name_mapping["HO5'"]],
        pmd_mol.atoms[atom_name_mapping["O5'"]]
    ]
    for atom in atom_delete:
        pmd_mol.delete_atom(atom)
    os.remove(input_file.parent/f"{residue_name}._amber.mol2")
    return pmd_mol


def build_residue_with_phosphate(input_file, atom_name_mapping, residue_name, charge):
    
    # align to template
    rd_mol = rd_load_file(input_file, removeHs=False)
    template = Chem.MolFromPDBFile(str(template_dir/"sugar_template.pdb"), removeHs=False)
    atom_maps = map_atoms(rd_mol, template)
    new_mol = flexible_align(
        rd_mol,
        template,
        atom_mapping=atom_maps,
        force_constant=100.0,
    )

    # for atom in new_mol.GetAtoms():
    #     print(atom.GetProp("_TriposPartialCharge"))
    #     partial_charge = float(atom.GetProp("_TriposPartialCharge")) 
    #     if partial_charge is None:
    #         raise ValueError(f"Partial charge for atom {atom.GetSymbol()} is not found")
    #     atom.SetProp('_GasteigerCharge', str(np.round(partial_charge, decimals=6)))
    # with Chem.SDWriter(str(input_file.parent/f"{residue_name}._aligned._bk.sdf")) as writer:
    #     for cid in range(new_mol.GetNumConformers()):
    #         writer.write(new_mol, confId=cid)
    _pmd_mol = pmd.load_file(str(input_file))
    conformer = new_mol.GetConformer()
    for i in range(new_mol.GetNumAtoms()):
        new_pos = conformer.GetAtomPosition(i)
        _pmd_mol.atoms[i].xx = new_pos.x
        _pmd_mol.atoms[i].xy = new_pos.y
        _pmd_mol.atoms[i].xz = new_pos.z
    _pmd_mol.save(str(input_file.parent/f"{residue_name}._aligned._bk.mol2"), overwrite=True)

    antechamber(
        input_file=input_file.parent/f"{residue_name}._aligned._bk.mol2", 
        input_type="mol2", 
        output=input_file.parent/f"{residue_name}._amber.mol2", 
        atom_type="amber", 
        residue_name=residue_name,
        charge=charge
    )
    pmd_mol = pmd.load_file(str(input_file.parent/f"{residue_name}._amber.mol2"))
    pmd_mol.fix_charges()
    
    pmd_mol.atoms[atom_name_mapping["O5'"]].charge = O5_charge
    pmd_mol.atoms[atom_name_mapping["O5'"]].type = "OS"
    pmd_mol.atoms[atom_name_mapping["O3'"]].charge = O3_charge
    pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OS"
    phosphate_mol = pmd.load_file(str(template_dir/"sugar_template_with_PO3.pdb"))
    phosphate_template = pmd.modeller.residue.ResidueTemplate.from_residue(phosphate_mol.residues[0])
    
    P = pmd.Atom(name="P", type="P", charge=P_charge, atomic_number=15)
    P.xx = phosphate_template.map["P"].xx
    P.xy = phosphate_template.map["P"].xy
    P.xz = phosphate_template.map["P"].xz
    OP1 = pmd.Atom(name="OP1", type="O2", charge=OP1_charge, atomic_number=8)
    OP1.xx = phosphate_template.map["OP1"].xx
    OP1.xy = phosphate_template.map["OP1"].xy
    OP1.xz = phosphate_template.map["OP1"].xz
    OP2 = pmd.Atom(name="OP2", type="O2", charge=OP2_charge, atomic_number=8)
    OP2.xx = phosphate_template.map["OP2"].xx
    OP2.xy = phosphate_template.map["OP2"].xy
    OP2.xz = phosphate_template.map["OP2"].xz

    pmd_mol.add_atom(P)
    pmd_mol.add_atom(OP1)
    pmd_mol.add_atom(OP2)
    pmd_mol.add_bond(pmd_mol.atoms[atom_name_mapping["O5'"]], pmd_mol.map["P"], 1)
    pmd_mol.add_bond(pmd_mol.map["P"], pmd_mol.map["OP1"], 1)
    pmd_mol.add_bond(pmd_mol.map["P"], pmd_mol.map["OP2"], 1)
    
    pmd_mol.tail = pmd_mol.atoms[atom_name_mapping["O3'"]]
    pmd_mol.head = pmd_mol.map["P"]
    atom_delete = [
        pmd_mol.atoms[atom_name_mapping["HO3'"]],
        pmd_mol.atoms[atom_name_mapping["HO5'"]],
    ]
    for atom in atom_delete:
        pmd_mol.delete_atom(atom)
    os.remove(input_file.parent/f"{residue_name}._aligned._bk.mol2")
    os.remove(input_file.parent/f"{residue_name}._amber.mol2")
    return pmd_mol


def build_residue_cap(input_file, atom_name_mapping, residue_name, charge):
    pmd_mol = pmd.load_file(input_file)
    return pmd_mol


def make_fragment(
        input_file,
        residue_name,
        residue_type="with_phosphate",
        output_dir=".",
        charge=0
        
    ):
    input_file = Path(input_file)
    residue_name = residue_name
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sugar_template_file = template_dir/"sugar_template.pdb"
    sugar_template = Chem.MolFromPDBFile(str(sugar_template_file), removeHs=False)
    rd_mol = rd_load_file(input_file, removeHs=False)
    atom_maps = map_atoms(sugar_template, rd_mol)
    mapping_dict = {}
    for atom in atom_maps:
        mapping_dict[atom[0]] = atom[1]
    name2idx = atomName_to_index(sugar_template)
    atom_name_mapping = {}
    for name, idx in name2idx.items():
        if idx in mapping_dict:
            atom_name_mapping[name] = mapping_dict[idx]

    if residue_type == "without_phosphate":
        pmd_mol = build_residue_without_phosphate(
                    input_file, 
                    atom_name_mapping,
                    residue_name,
                    charge=charge
                )
    elif residue_type == "with_phosphate":
        pmd_mol = build_residue_with_phosphate(
                    input_file, 
                    atom_name_mapping,
                    residue_name,
                    charge=charge
                )
    elif residue_type == "cap":
        pmd_mol = build_residue_cap(
                    input_file, 
                    atom_name_mapping,
                    residue_name,
                    charge=charge
                )
    else:
        raise ValueError("residue_type should be either 'without_phosphate', 'with_phosphate' or 'cap'")
    
    pmd_mol.save(str(output_dir/f"{residue_name}.mol2"), overwrite=True)
    pmd_mol.save(str(output_dir/f"{residue_name}.lib"), overwrite=True)
    pmd_mol.to_structure().write_pdb(str(output_dir/f"{residue_name}.pdb"), use_hetatoms=False)


def replace_residue(oligo_pdb, residue_pdb, residue_id, output_file):
    logger.info("Replacing residue %d in %s with %s", residue_id, oligo_pdb, residue_pdb)
    with open(oligo_pdb, 'r') as f:
        lines = f.readlines()
    
    old_residue_line_index = []
    for idx, line in enumerate(lines):
        if line[22:26].strip() == str(residue_id):
            old_residue_line_index.append(idx)
    
    index_diff = np.array(old_residue_line_index)[1:] - np.array(old_residue_line_index)[:-1]
    if not np.all(index_diff == 1):
        raise ValueError("Residue lines are not continuous")
    chain_id = lines[old_residue_line_index[0]][21]
    new_residue_lines = []
    with open(residue_pdb, 'r') as f:
        _new_lines = f.readlines()
        for line in _new_lines:
            _line = line[:22] + f"{residue_id:>4}" + line[26:]
            _line = _line[:21] + chain_id + _line[22:]  
            if not _line.endswith("\n"):
                _line += "\n"
            new_residue_lines.append(_line)
    
    del lines[old_residue_line_index[0]:old_residue_line_index[-1]+1]
    for line in new_residue_lines[::-1]:
        lines.insert(old_residue_line_index[0], line)
    logger.info("Writing to output file to %s", str(output_file))
    
    with open(output_file, 'w') as f:
        for idx, line in enumerate(lines):
            f.write(
                line[:6] + f"{idx+1:>5}" + line[11:]
            )