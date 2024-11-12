import os
from pathlib import Path


def read_charges(chg_file):
    charge_list = []
    with open(chg_file, "r") as fp:
        for line in fp.readlines():
            if len(line.strip()) < 1:
                continue
            charge_list.append(float(line.strip().split()[-1]))
    return charge_list

def GetCharges(wfn_file,  n_proc=1, log_file="multiwfn.log", workdir=None):
    multiwfn_script = ["7", "11", "1", "y", ]
    multiwfn_script += ["0", "0", "q"]

    cwd = Path(os.getcwd())
    if workdir is None:
        workdir = cwd
    with open(workdir/f"multiwfn_{Path(wfn_file).name}.sh", "w") as fp:
        fp.write(f"Multiwfn {wfn_file} -ispecial 1 -nt {n_proc} << EOF > {log_file}\n")
        fp.write("\n".join(multiwfn_script))
        fp.write("\nEOF")
        
    os.chdir(workdir)
    os.system(f"bash multiwfn_{Path(wfn_file).name}.sh")
    os.remove(f"multiwfn_{Path(wfn_file).name}.sh")
    os.chdir(cwd)
    