import sys
sys.path.append("/home/gridsan/ywang3/Project/Capping/parna/")
import parna
import parna.resp as resp
import parna.construct as construct
from pathlib import Path
import yaml
import argparse
import os
import shutil


def oligoprep(yaml_file):
    config_file = Path(yaml_file)
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input_pdb = Path(config["input_pdb"])
    charge_of_block = int(config["charge"])
    resname = str(config["resname"])
    res_output_dir = Path(config.get("lib_output_dir", f"lib/cap/{resname}.lib"))
    
    n_conformers = int(config["n_conformers"])
    skip_constraint = config["skip_constraint"]

    if not res_output_dir.exists():
        os.makedirs(res_output_dir, exist_ok=True)
    
    resp.generate_atomic_charges(
        input_files=str(input_pdb),
        charge=charge_of_block,
        output_dir=res_output_dir,
        residue_name=resname,
        skip_constraint=skip_constraint,
        scheme="resp2",
        memory="160 GB",
        n_threads=48,
        method_basis="wB97X-V/def2-TZVP",
        extra_charge_constraints=config.get("extra_charge_constraints", {}),
        extra_equivalence_constraints=config.get("extra_equivalence_constraints", []),
        charge_constrained_groups=["OH5"],
        prefix=resname+".resp2",
        nucleoside=True,
        generate_conformers=True,
        n_conformers=n_conformers
    )
    

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oligo parameterization")
    parser.add_argument("yaml_file", help="yaml file for oligo parameterization")
    args = parser.parse_args()
    oligoprep(args.yaml_file)







