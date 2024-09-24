import os
import subprocess
from pathlib import Path
from parna.logger import getLogger
import argparse

logger = getLogger(__name__)

class EngineORCA(object):
    def __init__(self):
        self.engine = "orca"
        self.orca_full_path = os.environ["ORCA_FULL_PATH"]

    def write_input(
            self,
            basis,
            method,
            xyz_file,
            solvent=None,
            n_proc=1,
            charge=0,
            orca_input_file="mol.inp"
    ):
        content = ["!MiniPrint"]
        content.append(f"!{basis} {method}")
        if solvent is not None:
            content.append(f"!CPCM({solvent})")
        content.append(f"%PAL NPROCS {n_proc} END")
        assert Path(xyz_file).suffix == ".xyz"
        content.append(f"* xyzfile {charge} 1 {xyz_file}")
        logger.info("\n".join(content))
        with open(orca_input_file, "w") as f:
            f.write("\n".join(content))
            f.write("\n")  # an extra blank line avoid error of ORCA
        logger.info(f"orca input file is written to {orca_input_file}")


    def run(self, input_file, job_path = None):
        cwd = os.getcwd()
        if job_path is not None:
            os.chdir(job_path)
        code = os.system(f"{self.orca_full_path} {input_file} --oversubscribe")
        os.chdir(cwd)
        return code
    
    def gwb2molden(self, input_file, output_file, job_path = None):
        cwd = os.getcwd()
        output_file = Path(output_file).resolve()
        if job_path is not None:
            os.chdir(job_path)
        input_file_stem = Path(input_file).stem
        code = os.system(f"orca_2mkl {input_file_stem} -molden")
        os.rename(f"{input_file_stem}.molden.input", output_file)
        os.chdir(cwd)
        return code


def calculate_energy_orca(
        input_file,
        output_dir, 
        charge=0, 
        n_threads=48, 
        method_basis="HF/6-31G*",
        aqueous=False,
    ):
    logger.info(f"calculating {method_basis} energy for " + str(input_file))
    output_dir = Path(output_dir)
    inFile = Path(input_file)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    orca = EngineORCA()
    orca_input = output_dir/f"{inFile.stem}.inp"
    method, basis = method_basis.split("/")
    if aqueous:
        solvent = "water"
    else:
        solvent = None
    orca.write_input(
        basis,
        method,
        str(inFile.resolve()),
        n_proc=n_threads,
        charge=charge,
        orca_input_file=str(orca_input.resolve()),
        solvent=solvent
    )
    code = orca.run(str(orca_input.resolve()), job_path=output_dir)
    if code != 0:
        logger.error(f"ORCA calculation failed for {input_file}")
        raise RuntimeError(f"ORCA calculation failed for {input_file}")
    orca.gwb2molden(
        str(inFile.stem)+".gwb",
        str(output_dir/f"{inFile.stem}.molden"),
        job_path=output_dir
    )
    return code



def read_energy_from_txt(log_file: str, source='orca') -> float:
    if source == 'psi4':
        with open(log_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if "Final Energy:" in line:
                energy = float(line.split()[-1])
                return energy
        return None
    elif source == 'orca':
        with open(log_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if "SCF Energy:" in line:
                energy = float(line.split()[-1])
                return energy
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--n_threads", type=int, default=48)
    parser.add_argument("--method_basis", type=str, default="HF/6-31G*")
    parser.add_argument("--aqueous", action="store_true")
    args = parser.parse_args()
    logger.info(f"Calculating energy by {__file__}")
    calculate_energy_orca(
        args.input_file,
        args.output_dir,
        charge=args.charge,
        n_threads=args.n_threads,
        method_basis=args.method_basis,
        aqueous=args.aqueous
    )