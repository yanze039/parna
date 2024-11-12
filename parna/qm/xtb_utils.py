import numpy as np
import shutil
import subprocess
from pathlib import Path
from parna.utils import getStringlist
from parna.logger import getLogger
import os

logger = getLogger(__name__)


logger.info("Checking existence of xtb...")
if not shutil.which("xtb"):
    raise FileNotFoundError("xtb is not found in the PATH. Please install xtb and add it to the PATH.")


def write_xtb_input(dihedral_atoms, dihedral_angles, scan_atoms, scan_type="dihedral", scan_start=0, scan_end=288, scan_steps=5, force_constant=0.05):
    """ Write xtb input file for dihedral scan.
        Input:
            dihedral_atoms: list of tuples, each tuple is a dihedral atom index
    """
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


def xtb(coord, inp=None, charge=0, workdir=".", solution="h2o", 
        sampling=False, gfnff=False,
        opt=True, opt_type=""):
    workdir = Path(workdir)
    coord = Path(coord)
    
    xtbcommand = [
        "xtb", str(coord.resolve()),  "--chrg", str(charge)
    ]
    if inp is not None:
        inp = Path(inp)
        xtbcommand.append("--input")
        xtbcommand.append(str(inp.resolve()))
    if opt:
        xtbcommand.append("--opt")
        xtbcommand.append(opt_type)
    if sampling:
        if gfnff:
            xtbcommand.append("--omd")
        else:
            xtbcommand.append("--md")
    if solution is not None:
        xtbcommand.append("--gbsa")
        xtbcommand.append(solution)
    if gfnff:
        xtbcommand.append("--gfnff")
    logger.info("Running xtb...")
    if not workdir.exists():
        workdir.mkdir(exist_ok=True)
    with open(workdir/"xtb.out", "w") as f:
        logger.info("Executing Command: " + " ".join(xtbcommand))
        result = subprocess.run(xtbcommand, stdout=f, stderr=f, cwd=str(workdir))
    if result.returncode != 0:
        logger.error("xtb failed.")
        logger.error("Please check the output file: " + str(workdir/"xtb.out"))
        raise RuntimeError("xtb failed.")
    else:
        logger.info("xtb Done.")


def structure_optimization(pdbfile, charge, restrain_atom_list=None, workdir="xtb"):
    if restrain_atom_list is not None:
        atom_string = ",".join([str(i) for i in restrain_atom_list])
        opt_inp = [
        "$fix",
        f"atoms: {atom_string}",
        "$end"
        ]
    else:
        opt_inp = []
    workdir = Path(workdir)
    if not workdir.exists():
        workdir.mkdir(exist_ok=True)
    with open(workdir/"xtb.inp", "w") as f:
        f.write("\n".join(opt_inp))
    xtb(
        coord=pdbfile,
        inp=(workdir/"xtb.inp").resolve(),
        charge=charge,
        workdir=workdir
    )
    

def gen_multi_conformations(input_file, charge, work_dir, ewin=6, threads=48):
    opt_command = ["xtb", str(Path(input_file).resolve()), "--opt", "--gbsa", "h2o", "--chrg", str(charge)]
    logger.info(" ".join(opt_command))
    subprocess.run(
        opt_command,
        cwd=work_dir,
    )
    subprocess.run(
        [
            "crest", 
            "xtbopt." + input_file.split(".")[-1],
            "-T", str(threads),
            "-g", "water", "-chrg", str(charge),
            "-ewin", str(ewin), "squick"
        ],
        cwd=work_dir,
    )
    return  


class XTB:
    def __init__(
        self, 
        force_constant: float = 0.05
    ):
        self.constraint = False
        self.constraint_list = []
        self.scan = False
        self.force_constant = force_constant
        self.scan_list = []
        self.sampling = False
    
    def clear(self):
        self.constraint = False
        self.constraint_list = []
        self.scan = False
        self.scan_list = []
    
    def clear_constraints(self):
        self.constraint = False
        self.constraint_list = []
    
    def set_force_constant(self, force_constant):
        self.force_constant = force_constant
    
    def add_constraint(self, constraint_type: str, atoms, at, force_constant=0.05):
        self.constraint = True
        self.set_force_constant(force_constant)
        self.constraint_list.append((constraint_type, atoms, at))
    
    def set_scan(self, scan_type, atoms, start, end, steps):
        self.scan = True
        self.scan_list.append((scan_type, atoms, start, end, steps))
    
    def write_constraint(self):
        contents = ["$constrain"]
        contents.append("force constant={}".format(str(self.force_constant)))
        for c in self.constraint_list:
            contents.append("{}: {},{}".format(c[0], ",".join(getStringlist([a+1 for a in c[1]])), str(c[2])))
        return "\n".join(contents)

    def write_scan(self):
        contents = ["$scan"]
        for s in self.scan_list:
            contents.append("{}: {},{};{},{},{}".format(
                s[0], 
                ",".join(getStringlist([a+1 for a in s[1]])),
                np.round(s[2], 2),
                np.round(s[2], 2),
                np.round(s[3], 2),
                int(s[4])
            ))
        return "\n".join(contents)
    
    def set_sampling(self, time=1.5, temp=300, step=4.0, shake=2):
        contents = ["$md"]
        contents.append(f"temp={temp}")
        contents.append(f"time={time}")
        contents.append("dump=100.0")
        contents.append(f"step={step}")
        contents.append("velo=false")
        contents.append("nvt=true")
        contents.append("hmass=4")
        contents.append(f"shake={shake:d}")
        contents.append("sccacc=2.0")
        contents.append("$end")
        self.sampling_list = contents
    
    def generate_input(self):
        contents = []
        if self.constraint:
            contents.append(self.write_constraint())
        if self.sampling:
            self.sampling_list = []
            if self.gfnff:
                self.set_sampling(time=3, temp=298, step=2.0, shake=0)
            else:
                self.set_sampling(temp=298, time=1.0)
            contents.append("\n".join(self.sampling_list))
        if self.scan:
            contents.append(self.write_scan())
        contents.append("$end")
        return "\n".join(contents)

    def write_input(self, filename):
        with open(filename, "w") as f:
            f.write(self.generate_input())
    
    def run(self, coord: str, charge: int, workdir, opt=True, 
            sampling=False,
            solution="h2o",
            gfnff=False,
            inp_name:str="xtb.inp"):
        workdir = Path(workdir)
        coord = Path(coord)
        inp = workdir/inp_name
        if sampling==True:
            self.sampling = True
        else:
            self.sampling = False
        
        if gfnff:
            self.gfnff = True
        else:
            self.gfnff = False
        
        self.write_input(inp)
        xtb(
            coord=coord,
            inp=inp,
            charge=charge,
            workdir=workdir,
            opt=opt,
            solution=solution,
            sampling=sampling,
            gfnff=gfnff
        )
        return workdir/"xtb.out"
    
    
        
        