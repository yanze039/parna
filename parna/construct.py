import parmed as pmd
import rdkit.Chem as Chem
import rdkit.Chem.rdDetermineBonds as rdDetermineBonds
from pathlib import Path
import os
import numpy as np
import random

from parna.utils import antechamber, atomName_to_index, map_atoms, rd_load_file, read_yaml, map_atoms_openfe
from parna.molops import flexible_align
from parna.molops import ortho_frame, rotate_conformer, translate_conformer
from parna.logger import getLogger

logger = getLogger(__name__)

# template_dir = Path(__file__).parent/"template"
TEMPLATE = Path(__file__).parent/"template"
FRAG = Path(__file__).parent/"fragments"
DATA = Path(__file__).parent/"data"
FRAME = Path(__file__).parent/"local_frame"


O3_charge = -0.5246
O5_charge = -0.4989
O3_charge_3p = -0.6541
HO3_charge_3p = 0.4376
P_charge = 1.1662
OP1_charge = -0.7760
OP2_charge = -0.7760


def get_base_attach_atoms():
    return read_yaml(DATA/"attach_atoms.yaml")

def build_backbone_block(
        baseFile,
        backboneFile,
        baseAttachPoint,
        backboneAttachPoint,
        bb_anchor_atom_pos,
        base_local_frame,
        backbone_local_frame,
    ):
    # read parameters
    # baseFile = FRAG / f"Base_{base_name}_hvy.pdb"
    # backboneFile = FRAG / f"N{backbone_position}_backbone.pdb"
    # attach_atoms = read_yaml(DATA/"attach_atoms.yaml")
    # bb_anchor_atom_pos = np.load(FRAME/f"N{backbone_position:d}_attach_point.npy")
    # base_local_frame = np.load(FRAME/f"{base_name}_frame.npy")
    # backbone_local_frame = np.load(FRAME/f"N{backbone_position:d}_frame.npy")
    
    # read PDB file
    base = Chem.MolFromPDBFile(str(baseFile), sanitize=False)
    backbone = Chem.MolFromPDBFile(str(backboneFile), sanitize=False)
    # get new ortho basis from local frame, this is because N is not perfectly sp2 hybridized (planar).
    base_basis = ortho_frame(base_local_frame[0], base_local_frame[1])
    backbone_basis = ortho_frame(backbone_local_frame[0], backbone_local_frame[1])
    # now rotate the base to align to backbone, eqivalent to two changes of basis.
    # in theory, we rotate M1 back and rotate to M2, so backbone_basis @ np.linalg.inv(base_basis) ,
    # but remember coordinates are row vectors, therefore we transpose the vector above to get:
    # np.linalg.inv(backbone_basis) @ (base_basis)
    final_transform = np.matmul( np.linalg.inv(backbone_basis), (base_basis) )
    rotate_conformer(base.GetConformer(), final_transform)
    base_anchor_atom_pos = np.array(base.GetConformer().GetAtomPosition(baseAttachPoint))
    # translate molecules to backbone
    tranlation_vec = bb_anchor_atom_pos - base_anchor_atom_pos
    translate_conformer(base.GetConformer(), tranlation_vec)
    # combine the two fragments, connecting bonds
    nucleoside = Chem.CombineMols(backbone,base)
    edmol = Chem.EditableMol(nucleoside)
    edmol.AddBond(
        backboneAttachPoint,
        baseAttachPoint+backbone.GetNumAtoms(),
        order=Chem.rdchem.BondType.SINGLE
    )
    finalmol = edmol.GetMol()
    Chem.SanitizeMol(finalmol)
    return finalmol


def build_backbone(
        sequence, 
        baseAttachAtoms,
        backboneAttachAtoms,
        chain_id="A", 
        cap_file=FRAG/"m7gppp.pdb",
        template_dir=TEMPLATE,
        local_frame_dir=FRAME,
        ):
    template_dir = Path(template_dir)
    if sequence[0] == "cap":
        print("The oligo uses capped structure.")
        template = Chem.MolFromPDBFile(str(TEMPLATE/"m7gppp.pdb"), removeHs=False)
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
    
    local_frame_dir = Path(local_frame_dir)
    for pos, basename in enumerate(sequence, start=1):
        print(f"Building residue {pos} with base {basename}")
        _basename = basename[0]
        rd_block = build_backbone_block(
                        baseFile=FRAG/f"Base_{_basename}_hvy.pdb",
                        backboneFile=template_dir/f"N{pos}_backbone.pdb",
                        baseAttachPoint=baseAttachAtoms[_basename],
                        backboneAttachPoint=backboneAttachAtoms["N"+str(pos)],
                        bb_anchor_atom_pos=np.load(local_frame_dir/f"N{pos}_attach_point.npy"),
                        base_local_frame=np.load(FRAME/f"{_basename}_frame.npy"),
                        backbone_local_frame=np.load(local_frame_dir/f"N{pos}_frame.npy"),
                    )
        pmd_block = pmd.rdkit.load_rdkit( rd_block )
        template = Chem.MolFromPDBFile(str(TEMPLATE/f"{basename}.pdb"), removeHs=False)
        atom_maps = map_atoms(template, rd_block)
        for template_i, block_j in atom_maps:
            atom = pmd_block.atoms[block_j]
            atom.name = template.GetAtoms()[template_i].GetPDBResidueInfo().GetName()
            myStructure.add_atom(
                atom, 
                resname=basename,
                resnum=pos,
                chain=chain_id
            )
    return myStructure


def build_residue_pdb_block(
        input_file, residue_name, template_residue, 
        atomtype="amber", 
        bondCompare=Chem.rdFMCS.BondCompare.CompareAny,
        residue_tail_idx=None, template_tail_idx=None, 
        residue_head_idx=None, template_head_idx=None,
        only_heavy_atoms=True, 
        atomCompare=Chem.rdFMCS.AtomCompare.CompareElements,
        fuzzy_matching=False,
        matchChiralTag=False,
        completeRingsOnly=True,
        use_openfe=False
    ):
    logger.info(f"Building residue {residue_name} from {input_file} using template {template_residue}")
    rd_mol = rd_load_file(str(input_file), removeHs=False, atomtype=atomtype)
    if not "." in str(template_residue):
        attach_atoms = read_yaml(DATA/"attach_atoms.yaml")
        rd_tmpl_block = build_backbone_block(
            baseFile=FRAG/f"Base_{template_residue[0]}_hvy.pdb",
            backboneFile=FRAG / f"N1_backbone.pdb",
            baseAttachPoint=attach_atoms[template_residue[0]],
            backboneAttachPoint=attach_atoms["N1"],
            bb_anchor_atom_pos=np.load(FRAME/f"N1_attach_point.npy"),
            base_local_frame=np.load(FRAME/f"{template_residue[0]}_frame.npy"),
            backbone_local_frame=np.load(FRAME/f"N1_frame.npy"),
        )
    else:
        if only_heavy_atoms:
            rd_tmpl_block = rd_load_file(str(template_residue), removeHs=True, sanitize=False)
        else:
            rd_tmpl_block = rd_load_file(str(template_residue), removeHs=False, sanitize=False)
    # sanitize the molecule to update the ring-info, otherwise the atom mapping will fail.
    # Not sanitize --> RuntimeError: Pre-condition Violation, RingInfo not initialized.
    Chem.SanitizeMol(rd_tmpl_block)

    editable_mol = Chem.EditableMol(rd_mol)
    editable_tmpl_mol = Chem.EditableMol(rd_tmpl_block)
    if residue_tail_idx is not None and template_tail_idx is not None:
        logger.info("Patching residue tail")
        new_atom_tail_idx = editable_mol.AddAtom(Chem.Atom(57))
        editable_mol.AddBond(residue_tail_idx, new_atom_tail_idx, Chem.BondType.DOUBLE)
        new_atom_tail_tmpl_idx = editable_tmpl_mol.AddAtom(Chem.Atom(57))
        editable_tmpl_mol.AddBond(template_tail_idx, new_atom_tail_tmpl_idx, Chem.BondType.DOUBLE)
    
    if residue_head_idx is not None and template_head_idx is not None:
        logger.info("Patching residue head")
        new_atom_head_idx = editable_mol.AddAtom(Chem.Atom(58))
        editable_mol.AddBond(residue_head_idx, new_atom_head_idx, Chem.BondType.SINGLE)
        new_atom_head_tmpl_idx = editable_tmpl_mol.AddAtom(Chem.Atom(58))
        editable_tmpl_mol.AddBond(template_head_idx, new_atom_head_tmpl_idx, Chem.BondType.SINGLE)
    
    rd_tmpl_block = editable_tmpl_mol.GetMol() 
    rd_mol = editable_mol.GetMol()

    print(">>> rd_mol:", rd_mol)
    print(">>> rd_tmpl_block:", rd_tmpl_block)
    print("completeRingsOnly:", completeRingsOnly)
    print("matchChiralTag:", matchChiralTag)
    print("fuzzy_matching:", fuzzy_matching)

    
    if use_openfe:
        atom_mapping = map_atoms_openfe(
            rd_tmpl_block, 
            rd_mol, 
        )
    else:
        atom_mapping = map_atoms(rd_tmpl_block, rd_mol, ringMatchesRingOnly=True, 
                                matchChiralTag=matchChiralTag,
                                completeRingsOnly=completeRingsOnly, 
                                atomCompare=atomCompare, 
                                bondCompare=bondCompare,
                                fuzzy_matching=fuzzy_matching)
    logger.info("Atom mapping:")
    logger.info(atom_mapping)
    logger.info(f"Head mapping: {template_head_idx} {residue_head_idx}, ")
    logger.info(f"Tail mapping: {template_tail_idx} {residue_tail_idx}, ")
    
    if residue_head_idx is not None and template_head_idx is not None:
        editable_mol.RemoveAtom((new_atom_head_idx))
        editable_tmpl_mol.RemoveAtom((new_atom_head_tmpl_idx))
        atom_mapping = [(i, j) for i, j in atom_mapping if i != new_atom_head_tmpl_idx]
    if residue_tail_idx is not None and template_tail_idx is not None:
        editable_mol.RemoveAtom((new_atom_tail_idx))
        editable_tmpl_mol.RemoveAtom((new_atom_tail_tmpl_idx))
        atom_mapping = [(i, j) for i, j in atom_mapping if i != new_atom_tail_tmpl_idx]
    
    rd_tmpl_block = editable_tmpl_mol.GetMol() 
    rd_mol = editable_mol.GetMol()

    logger.info(atom_mapping)
    
    inverse_atom_mapping = [(j, i) for i, j in atom_mapping]

    rd_mol = flexible_align(
        rd_mol,
        rd_tmpl_block,
        atom_mapping=inverse_atom_mapping,
        force_constant=500.0,
    )

    # Find BUG: the atom names are not preserved from mol2. we use parmed to fix this.
    # pdbblock = Chem.MolToPDBBlock(rd_mol)
    pmd_mol = pmd.rdkit.load_rdkit(rd_mol)
    old_mol = pmd.load_file(str(input_file))
    for idx, atom in enumerate(pmd_mol.atoms):
        atom.name = old_mol.atoms[idx].name
    tmp_pdb = f"{input_file.stem}_{residue_name}_{random.randint(100, 999)}_tmp.pdb"
    pmd_mol.write_pdb(tmp_pdb)
    with open(tmp_pdb, "r") as f:
        pdbblock = f.read()
    atom_list = []
    for line in pdbblock.split("\n"):
        if line.startswith("ATOM"):
            line = line[:17] + f"{residue_name:>3}" + line[20:]
            atom_list.append(line)
        elif line.startswith("HETATM"):
            line = "ATOM  " + line[6:17] + f"{residue_name:>3}" + line[20:]
            atom_list.append(line)
    
    line_order = []
    inverse_atom_mapping.sort(key=lambda x: x[0])
    for idx, (i, j) in enumerate(inverse_atom_mapping):
        line_order.append(i)
    
    for idx in range(len(atom_list)):
        if idx not in line_order:
            line_order.append(idx)
    atom_list = [atom_list[i] for i in line_order]
    os.remove(tmp_pdb)
    return "\n".join(atom_list)


def build_residue_without_phosphate(input_file, atom_name_mapping, 
                                    residue_name, charge, terminal=None,
                                    oxygen_type="dcase"
                                    ):
    # antechamber(
    #     input_file=input_file, 
    #     input_type=input_file.suffix[1:], 
    #     output=input_file.parent/f"{residue_name}._amber.mol2", 
    #     atom_type="amber", 
    #     residue_name=residue_name,
    #     charge=charge
    # )
    get_amber_mol2(
        input_file=input_file,
        output_file=input_file.parent/f"{residue_name}._amber.mol2",
        residue_name=residue_name,
        charge=charge,
    )
    pmd_mol = pmd.load_file(str(input_file.parent/f"{residue_name}._amber.mol2"))
    pmd_mol.fix_charges()
    
    if terminal == "3'":
        pmd_mol.atoms[atom_name_mapping["C5'"]].type = "CI"
        pmd_mol.atoms[atom_name_mapping["O3'"]].charge = O3_charge_3p
        pmd_mol.atoms[atom_name_mapping["HO3'"]].charge = HO3_charge_3p
        pmd_mol.head = pmd_mol.atoms[atom_name_mapping["C5'"]]
        atom_delete = [
            pmd_mol.atoms[atom_name_mapping["HO5'"]],
            pmd_mol.atoms[atom_name_mapping["O5'"]]
        ]
    elif terminal == "5'":
        pmd_mol.atoms[atom_name_mapping["O3'"]].charge = O3_charge
        # pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OS"
        if oxygen_type == "dcase":
            pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OR"
        else:
            pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OS"
        pmd_mol.tail = pmd_mol.atoms[atom_name_mapping["O3'"]]
        atom_delete = [
            pmd_mol.atoms[atom_name_mapping["HO3'"]],
        ]
    else:
        pmd_mol.atoms[atom_name_mapping["C5'"]].type = "CI"
        pmd_mol.atoms[atom_name_mapping["O3'"]].charge = O3_charge
        # pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OS"
        if oxygen_type == "dcase":
            pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OR"
        else:
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


def get_amber_mol2(input_file, output_file, residue_name, charge, check_dummy=True):
    input_file = Path(input_file)
    antechamber(
        input_file=str(input_file),
        input_type=input_file.suffix[1:],
        output=str(output_file),
        atom_type="amber", 
        residue_name=residue_name,
        charge=charge
    )
    if check_dummy:
        pmd_mol = pmd.load_file(str(output_file))
        for atom in pmd_mol.atoms:
            if atom.type == "DU" or atom.type == "NO" or atom.type == "N1":
                # raise ValueError("Dummy atom detected in the molecule.")
                tmp_gaff2_mol = output_file.parent/f"{residue_name}._gaff2.mol2"
                if not os.path.exists(tmp_gaff2_mol):
                    antechamber(
                        input_file=str(input_file),
                        input_type=input_file.suffix[1:],
                        output=str(tmp_gaff2_mol),
                        atom_type="gaff2", 
                        residue_name=residue_name,
                        charge=charge
                    )
                tmp_pmd_mol = pmd.load_file(str(tmp_gaff2_mol))
                atom.type = tmp_pmd_mol.map[atom.name].type
                os.remove(tmp_gaff2_mol)
        pmd_mol.save(str(output_file), overwrite=True)

def build_residue_with_phosphate(input_file, atom_name_mapping, 
                                 residue_name, charge, terminal=None,
                                 oxygen_type="dcase"
                                 ):
    
    # align to template
    rd_mol = rd_load_file(input_file, removeHs=False)
    template = Chem.MolFromPDBFile(str(TEMPLATE/"sugar_template.pdb"), removeHs=False)
    atom_maps = map_atoms(rd_mol, template, fuzzy_matching=False)
    # don't do fuzzy matching here, otherwise the alignment will have irregular conformers.
    new_mol = flexible_align(
        rd_mol,
        template,
        atom_mapping=atom_maps,
        force_constant=200.0,
    )

    _pmd_mol = pmd.load_file(str(input_file))
    conformer = new_mol.GetConformer()
    for i in range(new_mol.GetNumAtoms()):
        new_pos = conformer.GetAtomPosition(i)
        _pmd_mol.atoms[i].xx = new_pos.x
        _pmd_mol.atoms[i].xy = new_pos.y
        _pmd_mol.atoms[i].xz = new_pos.z
    _pmd_mol.save(str(input_file.parent/f"{residue_name}._aligned._bk.mol2"), overwrite=True)

    get_amber_mol2(
        input_file=input_file.parent/f"{residue_name}._aligned._bk.mol2",
        output_file=input_file.parent/f"{residue_name}._amber.mol2",
        residue_name=residue_name,
        charge=charge,
    )
    pmd_mol = pmd.load_file(str(input_file.parent/f"{residue_name}._amber.mol2"))
    pmd_mol.fix_charges()
    
    pmd_mol.atoms[atom_name_mapping["O5'"]].charge = O5_charge
    pmd_mol.atoms[atom_name_mapping["O5'"]].name = "O5'"
    pmd_mol.atoms[atom_name_mapping["C5'"]].type = "CI"

    phosphate_mol = pmd.load_file(str(TEMPLATE/"sugar_template_with_PO3.pdb"))
    phosphate_template = pmd.modeller.residue.ResidueTemplate.from_residue(phosphate_mol.residues[0])
    
    P = pmd.Atom(name="P", type="P", charge=P_charge, atomic_number=15)
    if oxygen_type == "dcase":
        OP1 = pmd.Atom(name="OP1", type="OP", charge=OP1_charge, atomic_number=8)
        OP2 = pmd.Atom(name="OP2", type="OP", charge=OP2_charge, atomic_number=8)
        pmd_mol.atoms[atom_name_mapping["O5'"]].type = "OR"
    else:
        OP1 = pmd.Atom(name="OP1", type="O2", charge=OP1_charge, atomic_number=8)
        OP2 = pmd.Atom(name="OP2", type="O2", charge=OP2_charge, atomic_number=8)
        pmd_mol.atoms[atom_name_mapping["O5'"]].type = "OS"
    P.xx = phosphate_template.map["P"].xx
    P.xy = phosphate_template.map["P"].xy
    P.xz = phosphate_template.map["P"].xz
    
    OP1.xx = phosphate_template.map["OP1"].xx
    OP1.xy = phosphate_template.map["OP1"].xy
    OP1.xz = phosphate_template.map["OP1"].xz
    
    OP2.xx = phosphate_template.map["OP2"].xx
    OP2.xy = phosphate_template.map["OP2"].xy
    OP2.xz = phosphate_template.map["OP2"].xz

    pmd_mol.add_atom(P)
    pmd_mol.add_atom(OP1)
    pmd_mol.add_atom(OP2)
    pmd_mol.add_bond(pmd_mol.atoms[atom_name_mapping["O5'"]], pmd_mol.map["P"], 1)
    pmd_mol.add_bond(pmd_mol.map["P"], pmd_mol.map["OP1"], 1)
    pmd_mol.add_bond(pmd_mol.map["P"], pmd_mol.map["OP2"], 1)
    
    pmd_mol.head = pmd_mol.map["P"]
    if terminal == "3'":
        pmd_mol.atoms[atom_name_mapping["O3'"]].charge = O3_charge_3p
        pmd_mol.atoms[atom_name_mapping["HO3'"]].charge = HO3_charge_3p
        pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OH"
        atom_delete = [
            pmd_mol.atoms[atom_name_mapping["HO5'"]],
        ]
    elif terminal == "5'":
        raise RuntimeError("5' terminal is not supported for phosphate")
    else:
        pmd_mol.atoms[atom_name_mapping["O3'"]].charge = O3_charge
        if oxygen_type == "dcase":
            pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OR"
        else:
            pmd_mol.atoms[atom_name_mapping["O3'"]].type = "OS"
        pmd_mol.tail = pmd_mol.atoms[atom_name_mapping["O3'"]]
        atom_delete = [
            pmd_mol.atoms[atom_name_mapping["HO3'"]],
            pmd_mol.atoms[atom_name_mapping["HO5'"]],
        ]
    for atom in atom_delete:
        pmd_mol.delete_atom(atom)
    os.remove(input_file.parent/f"{residue_name}._aligned._bk.mol2")
    os.remove(input_file.parent/f"{residue_name}._amber.mol2")
    
    return pmd_mol


def make_fragment(
        input_file,
        residue_name,
        residue_type="with_phosphate",
        output_dir=".",
        charge=0,
        suffix="",
        terminal=None,
        oxygen_type="dcase"
    ):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sugar_template_file = TEMPLATE/"sugar_template.pdb"
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
                    charge=charge,
                    terminal=terminal,
                    oxygen_type=oxygen_type
                )
    elif residue_type == "with_phosphate":
        pmd_mol = build_residue_with_phosphate(
                    input_file, 
                    atom_name_mapping,
                    residue_name,
                    charge=charge,
                    terminal=terminal,
                    oxygen_type=oxygen_type
                )
    else:
        raise ValueError("residue_type should be either 'without_phosphate', 'with_phosphate' or 'cap'")
    
    pmd_mol.save(str(output_dir/f"{residue_name}{suffix}.mol2"), overwrite=True)
    pmd_mol.save(str(output_dir/f"{residue_name}{suffix}.lib"), overwrite=True)
    pmd_mol.to_structure().write_pdb(str(output_dir/f"{residue_name}.pdb"), use_hetatoms=False)


def make_cap(
        input_file,
        residue_name,
        output_dir=".",
        charge=+1,
        suffix="",
    ):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sugar_template_file = TEMPLATE/"sugar_template.pdb"
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
     
    # antechamber(
    #     input_file=input_file, 
    #     input_type=input_file.suffix[1:], 
    #     output=input_file.parent/f"{residue_name}._amber.mol2", 
    #     atom_type="amber", 
    #     residue_name=residue_name,
    #     charge=charge
    # )
    get_amber_mol2(
        input_file=input_file,
        output_file=input_file.parent/f"{residue_name}._amber.mol2",
        residue_name=residue_name,
        charge=charge,
        check_dummy=True
    )
    pmd_mol = pmd.load_file(str(input_file.parent/f"{residue_name}._amber.mol2"))
    pmd_mol.fix_charges()
    pmd_mol.atoms[atom_name_mapping["C5'"]].type = "CI"
    pmd_mol.tail = pmd_mol.atoms[atom_name_mapping["C5'"]]
    atom_delete = [
        pmd_mol.atoms[atom_name_mapping["HO5'"]],
        pmd_mol.atoms[atom_name_mapping["O5'"]]
    ]
    for atom in atom_delete:
        pmd_mol.delete_atom(atom)
    os.remove(input_file.parent/f"{residue_name}._amber.mol2")
    pmd_mol.save(str(output_dir/f"{residue_name}{suffix}.mol2"), overwrite=True)
    pmd_mol.save(str(output_dir/f"{residue_name}{suffix}.lib"), overwrite=True)
    pmd_mol.to_structure().write_pdb(str(output_dir/f"{residue_name}.pdb"), use_hetatoms=False)
    return pmd_mol

def make_linker(
        input_file,
        residue_name,
        output_dir=".",
        charge=-3,
        suffix="",
        oxygen_type="dcase",
        head=None,
        tail=None
    ):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    rd_mol = rd_load_file(input_file, removeHs=False)
    pattern_mol = Chem.MolFromSmarts("C[*]P[*]P[*]P[*]C")
    pattern_atoms = rd_mol.GetSubstructMatches(pattern_mol)[0]
    
    P_atoms = [pattern_atoms[x] for x in [2, 4, 6]]
    OR_atoms = []
    for x in [1, 3, 5, 7]:
        atom = pattern_atoms[x]
        if rd_mol.GetAtomWithIdx(atom).GetAtomicNum() == 8:
            OR_atoms.append(atom)
    C_atoms = [pattern_atoms[0], pattern_atoms[-1]]
    OP_atoms = []
    for Pi in P_atoms:
        Patom = rd_mol.GetAtomWithIdx(Pi)
        neighbors = Patom.GetNeighbors()
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == 8 and neighbor.GetDegree() == 1:
                OP_atoms.append(neighbor.GetIdx())

    # antechamber(
    #     input_file=input_file, 
    #     input_type=input_file.suffix[1:], 
    #     output=input_file.parent/f"{residue_name}._amber.mol2", 
    #     atom_type="amber", 
    #     residue_name=residue_name,
    #     charge=charge
    # )
    get_amber_mol2(
        input_file=input_file,
        output_file=input_file.parent/f"{residue_name}._amber.mol2",
        residue_name=residue_name,
        charge=charge,
        check_dummy=False
    )
    
    pmd_mol = pmd.load_file(str(input_file.parent/f"{residue_name}._amber.mol2"))
    pmd_mol.fix_charges()
    
    if oxygen_type == "dcase":
        for ai in OP_atoms:
            pmd_mol.atoms[ai].type = "OP"
        for ai in OR_atoms:
            pmd_mol.atoms[ai].type = "OR"
    else:
        pass
    
    head = pattern_atoms[1] if head is None else head
    tail = pattern_atoms[-2] if tail is None else tail
    pmd_mol.tail = pmd_mol.atoms[tail]
    pmd_mol.head = pmd_mol.atoms[head]
    
    atom_delete = []
    for ci in C_atoms:
        atom_delete.append(ci)
        neighbors = rd_mol.GetAtomWithIdx(ci).GetNeighbors()
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == 1:
                atom_delete.append(neighbor.GetIdx())
        
    atom_delete.sort(reverse=True)
    atom_delete = [pmd_mol.atoms[ai] for ai in atom_delete]
    for atom in atom_delete:
        pmd_mol.delete_atom(atom)
        
    os.remove(input_file.parent/f"{residue_name}._amber.mol2")
    pmd_mol.save(str(output_dir/f"{residue_name}{suffix}.mol2"), overwrite=True)
    pmd_mol.save(str(output_dir/f"{residue_name}{suffix}.lib"), overwrite=True)
    pmd_mol.to_structure().write_pdb(str(output_dir/f"{residue_name}.pdb"), use_hetatoms=False)
    return pmd_mol

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


def make_fragment_cap(
        input_file,
        residue_name,
        output_dir,
        charge=-2
    ):
    """This is an old version for m7Gppp"""

    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    amber_mol2_file = output_dir/f"{residue_name}._amber.mol2"
    # antechamber(
    #     input_file=input_file, 
    #     input_type="mol2", 
    #     output=str(amber_mol2_file), 
    #     atom_type="amber", 
    #     residue_name=residue_name,
    #     charge=charge
    # )
    get_amber_mol2(
        input_file=input_file,
        output_file=amber_mol2_file,
        residue_name=residue_name,
        charge=charge
    )

    pmd_mol = pmd.load_file(str(amber_mol2_file))
    pmd_mol.fix_charges()

    OCH3_smarts = Chem.MolFromSmarts("PO[#6D4H3]")
    rd_mol  = rd_load_file(str(input_file), removeHs=False, atomtype="amber")
    OCH3_atoms = rd_mol.GetSubstructMatches(OCH3_smarts)
    terminal_O = OCH3_atoms[0][1]
    terminal_C = OCH3_atoms[0][2]
    CH3_Hs = []
    neighbors = rd_mol.GetAtoms()[terminal_C].GetNeighbors()
    for neighbor in neighbors:
        if neighbor.GetAtomicNum() == 1:
            CH3_Hs.append(neighbor.GetIdx())

    atom_delete = []
    for atom in pmd_mol.atoms:
        if atom.idx in CH3_Hs:
            atom_delete.append(atom)
        else:
            pass
    atom_delete.append(pmd_mol.atoms[terminal_C])
    new_tail = pmd_mol.atoms[terminal_O]

    for atom in atom_delete:
        pmd_mol.delete_atom(atom)

    pmd_mol.tail = new_tail
    pmd_mol.save(str(output_dir/f"{residue_name}.mol2"), overwrite=True)
    pmd_mol.save(str(output_dir/f"{residue_name}.lib"), overwrite=True)
    
    
def make_atom_name_mapping(
        input_file, template_residue,
        atomtype="amber", 
        bondCompare=Chem.rdFMCS.BondCompare.CompareOrder,
        residue_tail_idx=None, template_tail_idx=None, 
        residue_head_idx=None, template_head_idx=None,
        only_heavy_atoms=True, 
        atomCompare=Chem.rdFMCS.AtomCompare.CompareAnyHeavyAtom
    ):
    logger.info(f"mapping {input_file} using template {template_residue}")
    rd_mol = rd_load_file(str(input_file), removeHs=False, atomtype=atomtype)
    
    rd_tmpl_block = rd_load_file(str(template_residue), removeHs=only_heavy_atoms, 
                                 atomtype=atomtype,
                                 sanitize=False)
    # sanitize the molecule to update the ring-info, otherwise the atom mapping will fail.
    # Not sanitize --> RuntimeError: Pre-condition Violation, RingInfo not initialized.
    Chem.SanitizeMol(rd_tmpl_block)

    editable_mol = Chem.EditableMol(rd_mol)
    editable_tmpl_mol = Chem.EditableMol(rd_tmpl_block)
    if residue_tail_idx is not None and template_tail_idx is not None:
        new_atom_tail_idx = editable_mol.AddAtom(Chem.Atom(57))
        editable_mol.AddBond(residue_tail_idx, new_atom_tail_idx, Chem.BondType.DOUBLE)
        new_atom_tail_tmpl_idx = editable_tmpl_mol.AddAtom(Chem.Atom(57))
        editable_tmpl_mol.AddBond(template_tail_idx, new_atom_tail_tmpl_idx, Chem.BondType.DOUBLE)
    
    if residue_head_idx is not None and template_head_idx is not None:
        new_atom_head_idx = editable_mol.AddAtom(Chem.Atom(58))
        editable_mol.AddBond(residue_head_idx, new_atom_head_idx, Chem.BondType.SINGLE)
        new_atom_head_tmpl_idx = editable_tmpl_mol.AddAtom(Chem.Atom(58))
        editable_tmpl_mol.AddBond(template_head_idx, new_atom_head_tmpl_idx, Chem.BondType.SINGLE)
    
    rd_tmpl_block = editable_tmpl_mol.GetMol() 
    rd_mol = editable_mol.GetMol()
    
    atom_mapping = map_atoms(rd_tmpl_block, rd_mol, ringMatchesRingOnly=True, completeRingsOnly=True, 
                             atomCompare=atomCompare, bondCompare=bondCompare)
    logger.info("Atom mapping:")
    logger.info(atom_mapping)
    logger.info(f"{template_head_idx} {residue_head_idx}, ")
    logger.info(f"{template_tail_idx} {residue_tail_idx}, ")
    
    if residue_head_idx is not None and template_head_idx is not None:
        editable_mol.RemoveAtom((new_atom_head_idx))
        editable_tmpl_mol.RemoveAtom((new_atom_head_tmpl_idx))
        atom_mapping = [(i, j) for i, j in atom_mapping if i != new_atom_head_tmpl_idx]
    if residue_tail_idx is not None and template_tail_idx is not None:
        editable_mol.RemoveAtom((new_atom_tail_idx))
        editable_tmpl_mol.RemoveAtom((new_atom_tail_tmpl_idx))
        atom_mapping = [(i, j) for i, j in atom_mapping if i != new_atom_tail_tmpl_idx]

    logger.info(atom_mapping)
    mol_name2idx = atomName_to_index(rd_mol)
    mol_idx2name = {j:i for i, j in mol_name2idx.items()}
    tmpl_name2idx = atomName_to_index(rd_tmpl_block)
    tmpl_idx2name = {j:i for i, j in tmpl_name2idx.items()}
    
    atom_name_mapping = {}
    for i, j in atom_mapping:
        atom_name_mapping[tmpl_idx2name[i]] = mol_idx2name[j]
    return atom_name_mapping
    