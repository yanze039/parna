
import parna
from parna.construct import build_residue_pdb_block, replace_residue
from parna.parm import generate_frcmod
from parna.qm.xtb_utils import xtb
import rdkit.Chem as Chem
from pathlib import Path
import json
import parmed
import argparse
import os
import random




def make_residue(mol2_file, resname, residue_position, template_file, aligned_residue_pdb, input_oligo_file, output_oligo_file, lib_file=None):
    if residue_position  == 1:
        residue_pmd_mol = parmed.load_file(str(lib_file))[resname]
        residue_tail_idx = residue_pmd_mol.tail.idx
        pdb_block = build_residue_pdb_block(
            input_file=mol2_file, 
            residue_name=resname, 
            template_residue=template_file, 
            atomtype="amber",
            residue_tail_idx=residue_tail_idx, 
            template_tail_idx=1,
            atomCompare=Chem.rdFMCS.AtomCompare.CompareElements
        )
    elif residue_position == 2:
        residue_pmd_mol = parmed.load_file(str(lib_file))[resname]
        print(lib_file, residue_pmd_mol.tail)
        residue_tail_idx = residue_pmd_mol.tail.idx
        residue_head_idx = residue_pmd_mol.head.idx
        print(residue_head_idx, residue_tail_idx)
        print(residue_pmd_mol.head, residue_pmd_mol.tail)

        pdb_block = build_residue_pdb_block(
            input_file=mol2_file, 
            residue_name=resname, 
            template_residue=template_file, 
            atomtype="amber",
            residue_tail_idx=residue_tail_idx, 
            template_tail_idx=6,
            residue_head_idx=residue_head_idx,
            template_head_idx=2,
            atomCompare=Chem.rdFMCS.AtomCompare.CompareElements,
            bondCompare=Chem.rdFMCS.BondCompare.CompareAny
        )
    else:
        print(">>> else", resname, residue_position)
        pdb_block = build_residue_pdb_block(
            input_file=mol2_file, 
            residue_name=resname, 
            template_residue=template_file, 
            atomtype="amber",
            atomCompare=Chem.rdFMCS.AtomCompare.CompareElements,
            fuzzy_matching=False,
            only_heavy_atoms=False,
            matchChiralTag=False,
            completeRingsOnly=True,
            use_openfe=False
        )

    with open(aligned_residue_pdb, "w") as f:
        f.write(pdb_block)
    
    replace_residue(
        input_oligo_file,
        aligned_residue_pdb,
        residue_position,
        output_oligo_file
    )



def prep(seq: list, seq_name: str, is_generate_frcmod: bool = False):
    # flexible aligning 
    
    tmpl_oligo = parmed.load_file(str(oligo_pdb))
    oligoname = seq_name
    mod_oligo_file=output_oligo_dir/f"mod.oligo.{oligoname}.pdb"
    pmd_mol = parmed.load_file(str(oligo_pdb))
    n_residues = len(seq)
    pmd_mol = pmd_mol[f":1-{n_residues}"]
    pmd_mol.save(str(mod_oligo_file), use_hetatoms=False, overwrite=True)
    
    for pos, na in enumerate(seq):
        print(">>> ", na, pos)
        template_file = bk_output_dir/f"{tmpl_oligo.residues[pos].name}_{pos+1}.pdb"
        aligned_residue_pdb = bk_output_dir/f"{na}.{pos+1}.pdb"
        if na in lib_names:
            mol2_file = canonical_libs/f"{na}.mol2"
        else:
            if pos == 0:
                mol2_file = Path(lib_cap/f"cap/{na}.lib/{na}.mol2")
                lib_file = Path(lib_cap/f"cap/{na}.lib/{na}.lib")
            elif pos == 1:
                mol2_file = Path(lib_linker/f"linker/{na}.lib/{na}.mol2")
                lib_file = Path(lib_linker/f"linker/{na}.lib/{na}.lib")
            elif pos == 2:
                mol2_file = Path(lib_na/f"nucleotides/{na}.lib/{na}_without_phosphate.mol2")
                lib_file = Path(lib_na/f"nucleotides/{na}.lib/{na}.lib")
            else:
                mol2_file = Path(lib_na/f"nucleotides/{na}.lib/{na}_with_phosphate.mol2")
                lib_file = Path(lib_na/f"nucleotides/{na}.lib/{na}.lib")
        
        # pmd_mol = parmed.load_file(str(mol2_file))
        print(mol2_file)
        make_residue(
            mol2_file=mol2_file,
            resname=na,
            residue_position=pos+1,
            template_file=template_file,
            aligned_residue_pdb=aligned_residue_pdb,
            input_oligo_file=mod_oligo_file,
            output_oligo_file=mod_oligo_file,
            lib_file=lib_file
        )
    if is_generate_frcmod:
        try:
            generate_frcmod(
                input_file=mod_oligo_file,
                output_file=bk_output_dir/f"{oligoname}.frcmod",
                clean=False
            )
            assert os.path.exists(bk_output_dir/f"{oligoname}.frcmod")
            with open(bk_output_dir/f"{oligoname}.frcmod", "r") as fp:
                lines = list(fp.readlines())
            assert len(lines) > 20
        except Exception as e:
            print(e)
            tmp_dir = Path(f"tmp.{random.randint(0, 1000)}")
            tmp_dir.mkdir(exist_ok=False)
            print(tmp_dir)
            xtb(mod_oligo_file, 
                workdir=str(tmp_dir.resolve()), 
                charge=-5, 
                opt=True, 
                opt_type="crude", 
                # gfnff=True,
                sampling=False,
                solution=None,
                p=48,
                cycles=60
            )
            pdb_file = tmp_dir/f"xtbopt.pdb"
            
            generate_frcmod(
                input_file=pdb_file,
                output_file=bk_output_dir/f"{oligoname}.frcmod",
                clean=False
            )
            # shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="oligo parameterization")
    parser.add_argument("--sequence-components", "-sc", default="seq_comp.json", help="json file for sequence")
    parser.add_argument("--oligo-pdb", "-op", default="oligo.pdb", help="oligo pdb file")
    parser.add_argument("--backbone", "-b", default="backbone.lib", help="output directory")
    parser.add_argument("--output-dir", "-o", default="oligo", help="output directory")
    parser.add_argument("--generate-frcmod", "-f", action="store_true", help="generate frcmod file", default=False)
    args = parser.parse_args()
    
    # seq_json = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/seq_comp_doublelna_3na.json")
    # seq_json = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/seq_comp_5na.json")
    canonical_libs = Path("/home/gridsan/ywang3/Project/Capping/parna/parna/lib")
    lib_names = [x.stem for x in canonical_libs.glob("*.lib")]
    # oligo_pdb = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/oligo/eif4e3/eif4e3_ligand.pdb")
    # bk_output_dir = Path("backbone_eif4e3_oligo.lib")
    
    # oligo_pdb = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/oligo/eif4e1_binding_pose_1221/ligand.pdb")
    # bk_output_dir = Path("backbone_eif4e1_oligo_1221.lib")
    # output_oligo_dir = Path("oligo_1221")
    # output_oligo_dir.mkdir(exist_ok=True)
    
    # oligo_pdb = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/oligo/eif4e2_top/eif4e2_ligand_1222.pdb")
    # bk_output_dir = Path("backbone_eif4e2_oligo_1222.lib")
    # output_oligo_dir = Path("oligo_eif4e2_1222")
    # output_oligo_dir.mkdir(exist_ok=True)
    
    # oligo_pdb = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/oligo/6methyl_rep.c0.pdb")
    # bk_output_dir = Path("backbone_6_methyl_helix_1223.lib")
    # output_oligo_dir = Path("oligo_6_methyl_helix_1223")
    # output_oligo_dir.mkdir(exist_ok=True)
    
    seq_json = Path(args.sequence_components)
    oligo_pdb = Path(args.oligo_pdb)
    bk_output_dir = Path(args.backbone)
    if not os.path.exists(bk_output_dir):
        raise FileNotFoundError(f"{bk_output_dir} does not exist")
    output_oligo_dir = Path(args.output_dir)
    output_oligo_dir.mkdir(exist_ok=True)

    lib_na = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")
    lib_cap = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")
    lib_linker = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")
 
    with open(seq_json, "r") as f:
        seq_dict = json.load(f)
    for seq_name in seq_dict.keys():
        print(seq_name)
        prep(seq_dict[seq_name], seq_name=seq_name, is_generate_frcmod=args.generate_frcmod)










