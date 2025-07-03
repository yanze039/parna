import amberti
from amberti.workflow import setTI, submit_jobs
import json
import os
import shutil
from pathlib import Path
import numpy as np


lmb = np.linspace(0, 1, 30)
lmb_list = [np.round(l, 4) for l in lmb]


with open("./config.json", "r") as f:
    config = json.load(f)

# config["unified"]["lambdas"] = lmb_list
# config["np"] = len(lmb_list)
# config["ng"] = len(lmb_list)

# with open("./config.json", "w") as f:
#     json.dump(config, f, indent=4)

topology_dir = Path("./top")
base_dir = Path.cwd()
protocol = "unified"

for system in ["complex", "ligands"]:
    shutil.copy(topology_dir/f"{system}.parm7", topology_dir/f"{system}_{protocol}.parm7")
    shutil.copy(base_dir/f"{system}/pressurize.rst7", topology_dir/f"{system}_{protocol}.rst7")


tasks = setTI(
    config,
    top_dir="./top",
    protocol="unified",
    remd=True,
)
submit_jobs(
    tasks, 
    config['slurm_env'], 
    submit=False, 
    ngroup=-2, 
    n_lambda=len(config['unified']['lambdas'])
)


