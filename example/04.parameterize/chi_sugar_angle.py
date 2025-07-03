import parna
from parna.torsion.module import ChiTorsionFactory, SugarPuckerTorsionFactory   
import argparse
from pathlib import Path
import numpy as np
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_name", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--constrain-sugar", action="store_true", default=False)
    parser.add_argument("--charge", type=int, required=True)
    
    args = parser.parse_args()
    mol_name = args.mol_name
    cfactory = ChiTorsionFactory(
        order = 4, 
        panelty_weight= 0.1, 
        mol_name = mol_name,
        method = "wB97X-V",
        basis = "def2-TZVP",
        threads = 48,
        memory = "160 GB",
        resp_method= "wB97X-V",
        resp_basis = "def2-TZVP",
        resp_n_conformers = 6,  # 6
        output_dir = mol_name,
        cap_methylation=False,
        constrain_sugar = args.constrain_sugar,
        aqueous=True,
        fix_phase=True,
        pairwise=True,
    )
    cfactory.load_file(args.file, charge=args.charge)
    cfactory.gen(submit=False, local=True, overwrite=False)
    cfactory.optimize()
    
    spfactory = SugarPuckerTorsionFactory(
        order = 4, 
        panelty_weight= 0.1, 
        mol_name = mol_name,
        method = "wB97X-V",
        basis = "def2-TZVP",
        threads = 48,
        memory = "160 GB",
        resp_method= "wB97X-V",
        resp_basis = "def2-TZVP",
        resp_n_conformers = 6,  # 6
        output_dir = mol_name,
        cap_methylation=False,
        aqueous=True,
        fix_phase=True,
        pairwise=True,
    )
    spfactory.load_file(args.file, charge=args.charge)
    
    chi_conf_file = Path(mol_name) / "conformer_chi.yaml"
    chi_conf_data = spfactory.load_yaml(chi_conf_file)
    
    min_conf = None
    min_energy = int(1e6)
    for conf in chi_conf_data.keys():
        if chi_conf_data[conf]["qm_energy"] < min_energy:
            min_energy = chi_conf_data[conf]["qm_energy"]
            min_conf = conf
    dihedrals = chi_conf_data[min_conf]["dihedral"]
    print(dihedrals)
    print(f"Optimizing conformer {min_conf} with energy {min_energy}")
    angle_idx = list(dihedrals.keys())[0]
    angle_value = dihedrals[angle_idx]
    print(f"Optimizing angle {angle_idx} with value {angle_value}")
    
    spfactory.gen(submit=False, local=True, overwrite=False, 
                 extra_restraints={
                    "index": angle_idx,
                    "value": angle_value/np.pi*180,
                })
    
    spfactory.optimize()
    
    for i in range(5):
        
        parameter_set_chi = copy.deepcopy(cfactory.parameter_set)
        parameter_set_sp = copy.deepcopy(spfactory.parameter_set)
        parameter_set_sp["dihedral"] = [spfactory.valid_fragment["sugar-v1"], spfactory.valid_fragment["sugar-v3"]]
        parameter_set_chi["dihedral"] = list(cfactory.valid_fragment["fragment"].all_dihedral_quartets)
        
        cfactory.optimize(parameter_mod=parameter_set_sp, suffix=f"_{i:d}")
        spfactory.optimize(parameter_mod=parameter_set_chi, suffix=f"_{i:d}")
        
        print("Optimization round: ", i)
        print("Chi: ", cfactory.parameter_set)
        print("Sugar: ", spfactory.parameter_set)
    
    