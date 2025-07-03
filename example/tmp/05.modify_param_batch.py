import os
from pathlib import Path
import shutil
import argparse

parser = argparse.ArgumentParser(description="Modify parameters")
parser.add_argument("--fep-dir", help="FEP directory")
parser.add_argument("--parmpatch", help="parmpatch file")
# parser.add_argument("--backbone", help="backbone file", default="backbone.lib")
parser.add_argument("--nresidues-complex", help="number of residues in complex")
parser.add_argument("--nresidues-ligands", help="number of residues in ligands")
parser.add_argument("--capped", action="store_true", help="capped", default=False)
parser.add_argument("--alchem", action="store_true", help="normal", default=False)
args = parser.parse_args()

if args.capped:
    python_script = "05.modify_param_capped.py"
else:
    python_script = "05.modify_param_normal_strand.py"

if args.alchem:
    alchem = "--alchemical"
else:
    alchem = ""

command = "python {python_script} --parm7 {parm7} {alchem} --nresidues {nres}" \
    " --backbone {backbone} --output {output} --parmpatch {parmpatch}"
command = "python {python_script} --parm7 {parm7} {alchem} --nresidues {nres}" \
    " --output {output} --parmpatch {parmpatch}"

# FEP_dir = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/FEP.parna.5na.helix.1223")
FEP_dir = Path(args.fep_dir)
case_dirs = list(FEP_dir.glob("[A-Z]*_*"))

# case_dirs = [
#     Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/FEP.parna.5na.helix.0623.triplet/AGA_AAA")
# ]
# parmpatch = "/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/parm_patch_1204.yaml"
parmpatch = Path(args.parmpatch)
# os.system(command.format(parm7=test_file))
print(case_dirs)
for casedir in case_dirs:
    top_dir = casedir / "top"
    for mol in ["ligands", "complex"]:
        if mol == "ligands":
            nres = args.nresidues_ligands
        else:
            nres = args.nresidues_complex
        mycommand = command.format(
            python_script=python_script,
            parm7=top_dir / f"{mol}.parm7",
            # backbone=args.backbone,
            output=top_dir / f"{mol}.mod.parm7",
            parmpatch=parmpatch,
            alchem=alchem,
            nres=nres
        )
        try:
            exit_code = os.system(mycommand)
        # if exit_code != 0:
        #     print(f"Error in {top_dir / f'{mol}.parm7'}")
        #     raise SystemExit(exit_code)
            shutil.copy(top_dir / f"{mol}.rst7", top_dir / f"{mol}.mod.rst7")
        except Exception as e:
            print(f"Error in {top_dir / f'{mol}.parm7'}")
            print(e)
            continue
        

