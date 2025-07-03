import os
from pathlib import Path
import argparse
import json
import parmed
import sys
sys.path.append("/home/gridsan/ywang3/Project/Capping/parna")
import parna
from parna.construct import build_residue_pdb_block, replace_residue
from parna.parm import  parameterize, generate_frcmod
from parna.fep import get_softcore_region
from parna.utils import list2string
import rdkit
import rdkit.Chem as Chem
import shutil



def prep_fep_config(
        timask1, timask2, scmask1, scmask2,
        output_file
    ):
    template="/home/gridsan/ywang3/Project/Capping/oligo_simulation/screening/example/config.json"
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
        
        
component_json = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/seq_comp_5na.json")
protein_file = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data/oligo/eif4e1_eif4g_protein.pdb")
backbone_dir = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/backbone_eif4e1_eif4g_oligo.lib")
lib_na = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking/lib")
lib_cap = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")
lib_linker = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")

FEP_dir = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/binding_pose")
FEP_dir.mkdir(exist_ok=True)

linker_residues = ["M3N", "M3C"]

with open(component_json, "r") as f:
    component_dict = json.load(f)

def gen_topology(seq):
    seq_comp = component_dict[seq]
    
    
    fep_case_dir = FEP_dir / f"{seq}"
    fep_case_dir.mkdir(exist_ok=True)
    
    seq_oligo = backbone_dir / f"mod.oligo.{seq}.3na.pdb"
    
    n_residues = len(seq_comp)
    
    _tmp_pmd_mol = parmed.load_file(str(seq_oligo))
    _tmp_pmd_mol.strip(f"!(:1-{n_residues:d})")
    
    new_patch = []
    frcmod_list = [
        backbone_dir / f"{seq}.frcmod",
        "/home/gridsan/ywang3/Project/Capping/parna/parna/data/frcmod.99bsc0-chiol3-CaseP.frcmod"
    ]

    for i in range(len(seq_comp)):
        if i == 0:
            new_patch.append(lib_cap / f"cap/{seq_comp[i]}.lib/{seq_comp[i]}.lib")
        elif i == 1:
            new_patch.append(lib_linker / f"linker/{seq_comp[i]}.lib/{seq_comp[i]}.lib")
        elif i == 2:
            new_patch.append(lib_na / f"nucleotides_resp2/{seq_comp[i]}.lib/{seq_comp[i]}_without_phosphate.lib")
        else:
            new_patch.append(lib_na / f"nucleotides_resp2/{seq_comp[i]}.lib/{seq_comp[i]}_with_phosphate.lib")
    
    topology_dir = fep_case_dir / "top"

    # for complex-leg
    parameterize(
        oligoFile=seq_oligo,
        proteinFile=protein_file,
        external_libs=new_patch,
        additional_frcmods=frcmod_list,
        output_dir=topology_dir,
        n_cations=48,
        n_anions=48,
        prefix="complex",
        addons=[]
    )
            
            
       

if __name__ == "__main__":
    
    gen_topology("canonical_5na")
    
    
    
    
    
    