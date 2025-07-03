import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Modify parameters")
parser.add_argument("--fep-dir", help="FEP directory")
parser.add_argument("--parmpatch", help="parmpatch file")
parser.add_argument("--backbone", help="backbone file")
parser.add_argument("--nresidues", help="output directory")
args = parser.parse_args()

command = "python 05.modify_param_normal_strand.py --parm7 {parm7} --nresidues 4 --backbone {backbone}"

tetraNA_dir = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking/canonical_RNA_simulation/tetranucleotides")
case_dirs = [d for d in tetraNA_dir.iterdir() if d.is_dir()]


for casedir in case_dirs:
    top_dir = casedir / "parna"
    backbone_dir = top_dir / "backbone.lib"
    case_name = casedir.name
    parm_file = top_dir / f"{case_name}.mod.parm7"
    exit_code = os.system(command.format(
        parm7=parm_file,
        backbone=backbone_dir
    ))
    if exit_code != 0:
        print(f"Error in {parm_file}")
        raise SystemExit(exit_code)
        

