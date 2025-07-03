import os
from pathlib import Path
import yaml
import argparse


HEAD = """#!/bin/bash
source /etc/profile
module load anaconda/2023a
source activate mdtools
source $HOME/env/multiwfn.env
source $HOME/env/xtb.env
source $HOME/env/orca6.env

export OMP_NUM_THREADS=48
cd /home/gridsan/ywang3/Project/Capping/Benchmarking2
"""

parser = argparse.ArgumentParser(description="generate topology for double ligand")
parser.add_argument("pair_info_yaml", help="yaml file for pair info")
parser.add_argument('--sequence-components', "-sc", default="seq_comp.json", help="json file for sequence")
parser.add_argument('--protein', "-p", default="protein.pdb", help="protein pdb file")
parser.add_argument('--backbone-complex', "-bc", default="backbone.lib", help="backbone lib for complex")
parser.add_argument('--backbone-ligands', "-bl", default="backbone.lib", help="backbone lib for ligands")
parser.add_argument('--oligo-dir-complex', "-oc", default="oligo", help="oligo directory")
parser.add_argument('--oligo-dir-ligands', "-ol", default="oligo", help="oligo directory")
parser.add_argument('--fep-dir', "-f", default="FEP", help="FEP directory")
parser.add_argument("--residue-shift", "-rs", default=0, type=int, help="residue shift")
parser.add_argument("--capped",  action="store_true", help="force overwrite", default=False)
# take in additional lib paths, separated by space, take in as a list
parser.add_argument("--additional-libs", "-al", default=[], nargs="+", help="additional lib paths")
parser.add_argument("--additional-frcmods", "-af", default=[], nargs="+", help="additional lib paths")

args = parser.parse_args()

# pair_info_yaml = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/pair_doublelna_add.yaml")
pair_info_yaml = Path(args.pair_info_yaml)
with open(pair_info_yaml, "r") as f:
    pair_info = yaml.load(f, Loader=yaml.FullLoader)

submit_dir = Path("submit")
submit_dir.mkdir(exist_ok=True)
print(pair_info)
cwd = os.getcwd()

if args.capped:
    python_script = "04.gen_alchem_topology_with_cap.py"
else:
    python_script = "04.gen_alchem_topology.py"

for s1, s2 in pair_info["pairs"]:
    
    case_name = f"{s1}_{s2}"
    submit_file = submit_dir/f"{case_name}.sh"
    
    additional_libs = " ".join(args.additional_libs)
    additional_frcmods = " ".join(args.additional_frcmods)
    with open(submit_file, "w") as f:
        f.write(HEAD)
        f.write("\n")
        commands = f"python {python_script} {s1} {s2} " \
                  f"--sequence-components {args.sequence_components} " \
                  f"--protein {args.protein} " \
                  f"--backbone-complex {args.backbone_complex} " \
                  f"--backbone-ligands {args.backbone_ligands} " \
                  f"--oligo-dir-complex {args.oligo_dir_complex} " \
                  f"--oligo-dir-ligands {args.oligo_dir_ligands} " \
                  f"--residue-shift {args.residue_shift} " \
                  f"--fep-dir {args.fep_dir}\n"
        if len(args.additional_libs) > 0:
            commands += f" --addtional-libs {additional_libs} "
        if len(args.additional_frcmods) > 0:
            commands += f"--addtional-frcmods {additional_frcmods} " 
        f.write(commands)
    
    os.chdir(submit_dir)
    code = os.system(f"bash {submit_file.name} -s 4")
    if code != 0:
        print(f"Error in {case_name}, exit code: {code}")
        raise RuntimeError(f"Failed to generate topology for {case_name}")
    else:
        print(f"Successfully generated topology for {case_name}")
    os.chdir(cwd)


