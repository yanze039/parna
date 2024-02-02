import numpy as np
import subprocess
from pathlib import Path
from parna.utils import getStringlist
from parna.logger import getLogger

logger = getLogger()


def write_xtb_input(dihedral_atoms, dihedral_angles, scan_atoms, scan_type="dihedral", scan_start=0, scan_end=288, scan_steps=5, force_constant=0.05):
    contents = ["$constrain"]
    contents.append("force constant={}".format(str(force_constant)))
    for idx, diha in enumerate(dihedral_atoms):
        contents.append("dihedral: {},{}".format(",".join(getStringlist([a+1 for a in diha[:4]])), str(dihedral_angles[idx])))
    
    contents.append("$scan")
    contents.append("{}: {},{};{},{},{}".format(
        scan_type, 
        ",".join(getStringlist([a+1 for a in scan_atoms])),
        np.round(scan_start, 2),
        np.round(scan_start, 2),
        np.round(scan_end, 2),
        int(scan_steps)
    ))

    contents.append("$end")
    return "\n".join(contents)


def xtb(coord, inp, charge, workdir):
    workdir = Path(workdir)
    coord = Path(coord)
    inp = Path(inp)
    xtbcommand = [
        "xtb", str(coord.resolve()), "--input", str(inp.resolve()), "--chrg", str(charge), "--opt",
        "--gbsa", "h2o"
    ]
    logger.info("Running xtb...")
    if not workdir.exists():
        workdir.mkdir(exist_ok=True)
    with open(workdir/"xtb.out", "w") as f:
        logger.info("Executing Command: " + " ".join(xtbcommand))
        subprocess.run(xtbcommand, stdout=f, stderr=f, cwd=str(workdir))
    logger.info("xtb Done.")
    