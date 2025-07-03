import parna
from parna.torsion.module import EpsilonZetaTorsionFactory
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_name", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--constraint-sugar-pucker", action="store_true", default=False)
    
    args = parser.parse_args()
    mol_name = args.mol_name
    tfactory = EpsilonZetaTorsionFactory(
        order = 4, 
        panelty_weight= 0.1, 
        mol_name = mol_name,
        method = "wB97X-V",
        basis = "def2-TZVPD",
        threads = 48,
        memory = "160 GB",
        resp_method= "wB97X-V",
        resp_basis = "def2-TZVPD",
        resp_n_conformers = 6,  # 6
        output_dir = mol_name,
        cap_methylation=False,
        aqueous=True,
        fix_phase=False,
        pairwise=True,
        constrain_sugar_pucker=args.constraint_sugar_pucker,
        phosphate_style="dcase",
        alpha_gamma_style="bsc0",
    )
    tfactory.load_file(args.file, charge=0)
    tfactory.gen(submit=False, local=True, overwrite=False)
    tfactory.getCMAP(suffix="cmap", overwrite=True)
    