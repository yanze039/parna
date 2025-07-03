import parna
from parna.torsion.module import SugarEpsilonZetaTorsionFactory
from parna.torsion.optim import LinearCMAPSolver
import argparse
import numpy as np
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_name", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    
    args = parser.parse_args()
    mol_name = args.mol_name
    
    sugar_grids = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60])
    epsilon_grids = np.linspace(-180, 180, 25)[: -1]
    zeta_grids = np.linspace(-180, 180, 25)[: -1]
    
    tfactory = SugarEpsilonZetaTorsionFactory(
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
        phosphate_style="dcase",
        alpha_gamma_style="bsc0",
        epsilon_grids= epsilon_grids,
        zeta_grids= zeta_grids,
        sugar_grids= sugar_grids,
        train_epochs = 100,
        pretrained_model_path = "/home/gridsan/ywang3/Project/Capping/test_parameterization/test_NNP/refining/aimnet2_wb97m_0.pth"
    )
    tfactory.load_file(args.file, charge=0)
    tfactory.match_template()
    tfactory.gen_qm_geometry(submit=True, overwrite=False)
    tfactory.gen(submit=True, local=False, overwrite=False, \
        overwrite_wfn=False,
        n_submit_groups=7, training_fraction=0.06)
    tfactory.prepare_training_data()
    tfactory.train_nnp()
    tfactory.getCMAP(jitted_model_path=tfactory.jitted_model, 
                     overwrite=True,
                     mm_force_constant=500)
    cmap_file = tfactory.cmap_file 
    n_param = 24 * 24 + 24 * len(sugar_grids)
    lsover = LinearCMAPSolver(n_param)
    with open(cmap_file, "r") as f:
        energy_info = yaml.safe_load(f)
    lsover.fit(energy_info, mol_name)
    
    