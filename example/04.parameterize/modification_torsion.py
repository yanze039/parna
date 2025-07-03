import parna
from parna.torsion.module import NonCanonicalTorsionFactory
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_name", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--charge", type=int, default=0)
    
    args = parser.parse_args()
    mol_name = args.mol_name
    tfactory = NonCanonicalTorsionFactory(
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
        fix_phase=False,
        pairwise=True,
    )
    tfactory.load_file(args.file, charge=args.charge)
    tfactory.gen(submit=False, local=True, overwrite=False)
    tfactory.optimize()
    