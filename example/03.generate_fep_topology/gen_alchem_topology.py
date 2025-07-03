import os
from pathlib import Path
import argparse
import json
import parmed
import parna
from parna.construct import build_residue_pdb_block, replace_residue
from parna.parm import alchemical_parameterize, parameterize, generate_frcmod
from parna.fep import get_softcore_region
from parna.utils import list2string
import rdkit
import rdkit.Chem as Chem
import shutil
from parna.qm.xtb_utils import xtb



def prep_fep_config(
        timask1, timask2, scmask1, scmask2,
        output_file
    ):
    template="fep_template/config.json"
    with open(template, "r") as f:
        config = json.load(f)
    config["prep"]["timask1"] = timask1
    config["prep"]["timask2"] = timask2
    config["prep"]["scmask1"] = scmask1
    config["prep"]["scmask2"] = scmask2
    config["prep"]["pressurize"]["nsteps"] = 2500000
    config["unified"]["timask1"] = timask1
    config["unified"]["timask2"] = timask2
    config["unified"]["scmask1"] = scmask1
    config["unified"]["scmask2"] = scmask2
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)
        
  

def strip_mol(input_file, output_file, n_residues):
    _tmp_pmd_mol = parmed.load_file(str(input_file))
    _tmp_pmd_mol.strip(f"!(:1-{n_residues:d})")
    _tmp_pmd_mol.write_pdb(str(output_file))
    

def gen_topology(seq1, seq2):
    seq1_comp = component_dict[seq1]
    seq2_comp = component_dict[seq2]
    
    
    fep_case_dir = FEP_dir / f"{seq1}_{seq2}"
    fep_case_dir.mkdir(exist_ok=True)
    
    seq1_oligo_complex = oligo_struct_dir_complex / f"mod.oligo.{seq1}.pdb"
    seq2_oligo_complex = oligo_struct_dir_complex / f"mod.oligo.{seq2}.pdb"
    seq1_oligo_ligands = oligo_struct_dir_ligands / f"mod.oligo.{seq1}.pdb"
    seq2_oligo_ligands = oligo_struct_dir_ligands / f"mod.oligo.{seq2}.pdb"
    
    aligned_seq1_oligo_complex = fep_case_dir / f"mod.oligo.{seq1}.complex.pdb"
    aligned_seq2_oligo_complex = fep_case_dir / f"mod.oligo.{seq2}.complex.pdb"
    aligned_seq1_oligo_ligands = fep_case_dir / f"mod.oligo.{seq1}.ligands.pdb"
    aligned_seq2_oligo_ligands = fep_case_dir / f"mod.oligo.{seq2}.ligands.pdb"
    
    n_residues = len(seq1_comp)

    strip_mol(seq1_oligo_complex, aligned_seq1_oligo_complex, n_residues)
    strip_mol(seq2_oligo_complex, aligned_seq2_oligo_complex, n_residues)
    strip_mol(seq1_oligo_ligands, aligned_seq1_oligo_ligands, n_residues)
    strip_mol(seq2_oligo_ligands, aligned_seq2_oligo_ligands, n_residues)
    
    for pos in range(len(seq2_comp)):
        seq1_res = seq1_comp[pos]
        seq2_res = seq2_comp[pos]
        
        pos_idx = pos + args.residue_shift
        
        seq1_res_pdb_complex = backbone_dir_complex / f"{seq1_res}.{pos_idx+1}.pdb"  # template
        seq1_res_pdb_ligands = backbone_dir_ligands / f"{seq1_res}.{pos_idx+1}.pdb"  # template
        
        if pos == 0:
            seq2_res_mol2 = Path(lib_na/f"nucleotides/{seq2_res}.lib/{seq2_res}_without_phosphate.mol2")
            seq2_res_lib = Path(lib_na/f"nucleotides/{seq2_res}.lib/{seq2_res}_without_phosphate.lib")
        else:
            seq2_res_mol2 = Path(lib_na/f"nucleotides/{seq2_res}.lib/{seq2_res}_with_phosphate.mol2")
            seq2_res_lib = Path(lib_na/f"nucleotides/{seq2_res}.lib/{seq2_res}_with_phosphate.lib")
        
        seq2_res_pdb_complex = backbone_dir_complex / f"{seq2_res}.{pos_idx+1}.pdb"  # template
        seq2_res_pdb_ligands = backbone_dir_ligands / f"{seq2_res}.{pos_idx+1}.pdb"  # template
        
        if pos == 0:
            seq1_res_lib = Path(lib_cap/f"nucleotides/{seq1_res}.lib/{seq1_res}_without_phosphate.lib")
            seq2_res_lib = Path(lib_cap/f"nucleotides/{seq2_res}.lib/{seq2_res}_without_phosphate.lib")
            
            seq1_res_pmd_mol = parmed.load_file(str(seq1_res_lib))[seq1_res]
            seq1_tail_idx = seq1_res_pmd_mol.tail.idx
            seq2_res_pmd_mol = parmed.load_file(str(seq2_res_lib))[seq2_res]
            seq2_tail_idx = seq2_res_pmd_mol.tail.idx
            
            pdb_block_complex = build_residue_pdb_block(
                input_file=seq2_res_mol2, 
                residue_name=seq2_res, 
                template_residue=seq1_res_pdb_complex, 
                atomtype="amber",
                residue_tail_idx=seq2_tail_idx, 
                template_tail_idx=seq1_tail_idx,
                atomCompare=Chem.rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                only_heavy_atoms=True,
                fuzzy_matching=False
            )
            pdb_block_ligands = build_residue_pdb_block(
                input_file=seq2_res_mol2,
                residue_name=seq2_res,
                template_residue=seq1_res_pdb_ligands,
                atomtype="amber",
                residue_tail_idx=seq2_tail_idx,
                template_tail_idx=seq1_tail_idx,
                atomCompare=Chem.rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                only_heavy_atoms=True,
                fuzzy_matching=False
            )
        else:
            pdb_block_complex = build_residue_pdb_block(
                input_file=seq2_res_mol2, 
                residue_name=seq2_res, 
                template_residue=seq1_res_pdb_complex, 
                atomtype="amber",
                atomCompare=Chem.rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                only_heavy_atoms=True,
                fuzzy_matching=False
            )
            pdb_block_ligands = build_residue_pdb_block(
                input_file=seq2_res_mol2, 
                residue_name=seq2_res, 
                template_residue=seq1_res_pdb_ligands, 
                atomtype="amber",
                atomCompare=Chem.rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                only_heavy_atoms=True,
                fuzzy_matching=False
            )
        
        aligned_seq2_res_pdb_complex = fep_case_dir / f"{seq2_res}.{pos+1}.complex.pdb"
        aligned_seq2_res_pdb_ligands = fep_case_dir / f"{seq2_res}.{pos+1}.ligands.pdb"
        with open(aligned_seq2_res_pdb_complex, "w") as f:
            f.write(pdb_block_complex)
        with open(aligned_seq2_res_pdb_ligands, "w") as f:
            f.write(pdb_block_ligands)
        shutil.copy(seq1_res_pdb_complex, fep_case_dir/f"{seq1_res}.{pos+1}.complex.pdb")
        shutil.copy(seq1_res_pdb_ligands, fep_case_dir/f"{seq1_res}.{pos+1}.ligands.pdb")
        
        replace_residue(
            aligned_seq2_oligo_complex, aligned_seq2_res_pdb_complex,
            pos+1, aligned_seq2_oligo_complex )
        replace_residue(
            aligned_seq2_oligo_ligands, aligned_seq2_res_pdb_ligands,
            pos+1, aligned_seq2_oligo_ligands )
    
    oligoFile1_complex = fep_case_dir / f"mod.oligo.{seq1}.complex.pdb"
    oligoFile2_complex = fep_case_dir / f"mod.oligo.{seq2}.complex.pdb"
    oligoFile1_ligands = fep_case_dir / f"mod.oligo.{seq1}.ligands.pdb"
    oligoFile2_ligands = fep_case_dir / f"mod.oligo.{seq2}.ligands.pdb"
    
    new_patch = args.addtional_libs
    frcmod_list = args.addtional_frcmods
    frcmod_list += [
        backbone_dir_complex / f"{seq1}.frcmod",
        backbone_dir_complex / f"{seq2}.frcmod"
    ]

    for i in range(len(seq1_comp)):
        if i == 0:
            new_patch.append(lib_na / f"nucleotides/{seq1_comp[i]}.lib/{seq1_comp[i]}_without_phosphate.lib")
            new_patch.append(lib_na / f"nucleotides/{seq2_comp[i]}.lib/{seq2_comp[i]}_without_phosphate.lib")
        else:
            new_patch.append(lib_na / f"nucleotides/{seq1_comp[i]}.lib/{seq1_comp[i]}_with_phosphate.lib")
            new_patch.append(lib_na / f"nucleotides/{seq2_comp[i]}.lib/{seq2_comp[i]}_with_phosphate.lib")
    
    topology_dir = fep_case_dir / "top"
    
    
    # for complex-leg
    alchemical_parameterize(
        oligoFile1=oligoFile1_complex, 
        oligoFile2=oligoFile2_complex, 
        proteinFile=protein_file, 
        external_libs=new_patch, 
        additional_frcmods=frcmod_list,
        output_dir=topology_dir, 
        n_cations=12+14,
        n_anions=12,
        prefix="complex", 
        check_atomtypes=True 
    )

    # for ligands-leg
    alchemical_parameterize(
        oligoFile1=oligoFile1_ligands, 
        oligoFile2=oligoFile2_ligands, 
        proteinFile=None, 
        external_libs=new_patch, 
        additional_frcmods=frcmod_list,
        output_dir=topology_dir, 
        n_cations=9+7,
        n_anions=9, 
        prefix="ligands",
        check_atomtypes=True 
    )
    
    different_idx = None
    for i in range(len(seq1_comp)):
        if seq1_comp[i] != seq2_comp[i]:
            different_idx = i
            break
    if different_idx is None:
        raise RuntimeError("alchemical error.")
    seq1_res = seq1_comp[different_idx]
    seq2_res = seq2_comp[different_idx]
    
    template_residue = Chem.MolFromPDBFile(str(fep_case_dir/f"{seq1_res}.{different_idx+1}.complex.pdb"), removeHs=False)
    query_residue = Chem.MolFromPDBFile(str(fep_case_dir/f"{seq2_res}.{different_idx+1}.complex.pdb"), removeHs=False)
    on_linker = seq1_res in linker_residues or seq2_res in linker_residues
    query_unique_atom_names, template_unique_atom_names = get_softcore_region(query_residue, 
                                                                              template_residue, 
                                                                              return_atom_name=True, 
                                                                              on_linker=False)
    n_residues = len(seq1_comp)
    timask1 = f":1-{n_residues:d}"
    timask2 = f":{n_residues+1:d}-{n_residues*2:d}"
    if len(template_unique_atom_names) > 0:
        scmask1 = f":{different_idx+1}@{list2string(template_unique_atom_names)}"
    else:
        scmask1 = ""
    if len(query_unique_atom_names) > 0:
        scmask2 = f":{different_idx+1+n_residues}@{list2string(query_unique_atom_names)}"
    else:
        scmask2 = ""

    print(f"timask1: {timask1}")
    print(f"timask2: {timask2}")
    print(f"scmask1: {scmask1}")
    print(f"scmask2: {scmask2}")
        
    
    prep_fep_config(
        timask1, timask2, scmask1, scmask2,
        output_file=fep_case_dir/"config.json"
    )
    prep_script = Path("example/prepFEP.py")
    run_script = Path("example/runFEP.py")
    shutil.copy(prep_script, fep_case_dir/Path(prep_script.name))
    shutil.copy(run_script, fep_case_dir/Path(run_script.name))
    print("File saved to ", fep_case_dir)

    # submit job:
    with open(fep_case_dir/"submit.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("source /etc/profile\n")
        f.write("source $HOME/env/amber22.env\n")
        f.write(f"cd {fep_case_dir.resolve()}\n")
        f.write("python prepFEP.py config.json\n")
        f.write("python runFEP.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torsion scan')
    parser.add_argument('seq1', type=str, help='seq1')
    parser.add_argument('seq2', type=str, help='seq2')
    parser.add_argument('--sequence-components', "-sc", default="seq_comp.json", help="json file for sequence")
    parser.add_argument('--protein', "-p", default="protein.pdb", help="protein pdb file")
    parser.add_argument('--backbone-complex', "-bc", default="backbone.lib", help="backbone lib for complex")
    parser.add_argument('--backbone-ligands', "-bl", default="backbone.lib", help="backbone lib for ligands")
    parser.add_argument('--oligo-dir-complex', "-oc", default="oligo", help="oligo directory")
    parser.add_argument('--oligo-dir-ligands', "-ol", default="oligo", help="oligo directory")
    parser.add_argument('--fep-dir', "-f", default="FEP", help="FEP directory")
    parser.add_argument("--residue-shift", "-rs", default=0, type=int, help="residue shift")
    parser.add_argument("--capped",  action="store_true", help="force overwrite", default=False)
    parser.add_argument("--addtional-libs", "-al", default=[], nargs="+", help="additional lib paths")
    parser.add_argument("--addtional-frcmods", "-af", default=[], nargs="+", help="additional lib paths")
    
    args = parser.parse_args()
        
    component_json = Path("component.json")
    component_json = Path(args.sequence_components)
    protein_file = Path(args.protein)
    backbone_dir_complex = Path(args.backbone_complex)
    backbone_dir_ligands = Path(args.backbone_ligands)
    oligo_struct_dir_complex = Path(args.oligo_dir_complex)
    oligo_struct_dir_ligands = Path(args.oligo_dir_ligands)
    FEP_dir = Path(args.fep_dir)

    lib_na = Path("lib")
    lib_cap = Path("lib")
    lib_linker = Path("lib")

    FEP_dir.mkdir(exist_ok=True)

    linker_residues = ["M3N", "M3C"]

    with open(component_json, "r") as f:
        component_dict = json.load(f)
        
    
    gen_topology(args.seq1, args.seq2)
    

    
    