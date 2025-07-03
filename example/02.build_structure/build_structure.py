
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




def make_residue(mol2_file, resname, residue_position, template_file, aligned_residue_pdb, input_oligo_file, output_oligo_file):
    pdb_block = build_residue_pdb_block(
        input_file=mol2_file, 
        residue_name=resname, 
        template_residue=template_file, 
        atomtype="amber",
        atomCompare=Chem.rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        fuzzy_matching=True
    )

    with open(aligned_residue_pdb, "w") as f:
        f.write(pdb_block)
    
    replace_residue(
        input_oligo_file,
        aligned_residue_pdb,
        residue_position,
        output_oligo_file
    )



def prep(seq, seq_name: str):
    # flexible aligning 
    
    tmpl_oligo = parmed.load_file(str(oligo_pdb))
    oligoname = seq_name
    mod_oligo_file=output_oligo_dir/f"mod.oligo.{oligoname}.pdb"
    pmd_mol = parmed.load_file(str(oligo_pdb))
    n_residues = len(seq)
    pmd_mol = pmd_mol[f":1-{n_residues}"]
    pmd_mol.save(str(mod_oligo_file), use_hetatoms=False, overwrite=True)
    
        
    for pos, na in enumerate(seq):
        if args.double_stranded:
            pos_idx = int(pos%(len(seq)//2))
        else:
            pos_idx = pos + args.residue_shift
        template_file = bk_output_dir/f"{tmpl_oligo.residues[pos].name}_{pos_idx+1}.pdb"
        aligned_residue_pdb = bk_output_dir/f"{na}.{pos+1}.pdb"
        
        if pos == 0 or na[-1] == '5':
            mol2_file = Path(lib_na/f"nucleotides/{na}.lib/{na}_without_phosphate.mol2")
        else:
            mol2_file = Path(lib_na/f"nucleotides/{na}.lib/{na}_with_phosphate.mol2")
            
        
        make_residue(
            mol2_file=mol2_file,
            resname=na,
            residue_position=pos+1,
            template_file=template_file,
            aligned_residue_pdb=aligned_residue_pdb,
            input_oligo_file=mod_oligo_file,
            output_oligo_file=mod_oligo_file,
        )
    
    try:
        generate_frcmod(
            input_file=mod_oligo_file,
            output_file=bk_output_dir/f"{oligoname}.frcmod",
            clean=False
        )
        assert os.path.exists(bk_output_dir/f"{oligoname}.frcmod")
    except Exception as e:
        print(e)
        tmp_dir = Path(f"tmp.{random.randint(0, 1000)}")
        print(f"Using temporary directory {tmp_dir}")
        xtb(mod_oligo_file, workdir=str(tmp_dir.resolve()), charge=-3, opt=True, opt_type="crude")
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
    parser.add_argument("--double-stranded", "-ds", action="store_true", help="double stranded oligo", default=False)
    parser.add_argument("--residue-shift", "-rs", type=int, default=0, help="residue shift")
    args = parser.parse_args()
    
    canonical_libs = Path("/home/gridsan/ywang3/Project/Capping/parna/parna/lib")
    lib_names = [x.stem for x in canonical_libs.glob("*.lib")]
    oligo_pdb = Path(args.oligo_pdb)
    bk_output_dir = Path(args.backbone)

    lib_na = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")
    lib_cap = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")
    lib_linker = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")

    output_oligo_dir = Path(args.output_dir)
    output_oligo_dir.mkdir(exist_ok=True)
    seq_json = Path(args.sequence_components)
    
    with open(seq_json, "r") as f:
        seq_dict = json.load(f)
    for seq_name in seq_dict.keys():
        print(seq_name)
        prep(seq_dict[seq_name], seq_name=seq_name)











