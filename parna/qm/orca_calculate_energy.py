import os
import subprocess
from pathlib import Path
import logging
import argparse
import pickle

logger = logging.getLogger(__name__)

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
            opt=False,
            n_proc=1,
            charge=0,
            engrad=True,
            orca_input_file="mol.inp",
            constraints=None
    ):
        content = ["!MiniPrint"]
        content.append(f"!{basis} {method}")
        if opt:
            content.append("!OPT")
        if engrad:
            content.append("!ENGRAD")
        if solvent is not None:
            if "xtb" in method.lower():
                content.append(f"!DDCOSMO({solvent})")
            else:
                content.append(f"!CPCM({solvent})")
            
        content.append(f"%PAL NPROCS {n_proc} END")
        content.append("%Method")
        content.append("WriteJSONPropertyfile True")
        content.append("End")
        
        if constraints is not None:
            content.append(r"%geom")
            content.append("Constraints")
            for constraint in constraints:
                dihedral_indices = " ".join(map(str, constraint[0]))
                values = (float(constraint[1]) + 360) % 360
                content.append(f"{{ D {dihedral_indices} {values} C }}")
            content.append("end\nend")
        
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
        code = os.system(f"{self.orca_full_path} {input_file} --oversubscribe > {input_file}.orca.log")
        if code != 0:
            logger.error(f"ORCA calculation failed for {input_file}")
            raise RuntimeError(f"ORCA calculation failed for {input_file}")
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
        method="HF",
        basis="6-31G*",
        aqueous=False,
        opt=False,
        convert2molden=True,
        constraints=None,
        engrad=True
    ):
    logger.info(f"calculating {method}/{basis} energy for " + str(input_file))
    output_dir = Path(output_dir)
    inFile = Path(input_file)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    orca = EngineORCA()
    orca_input = output_dir/f"{inFile.stem}.inp"
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
        solvent=solvent,
        opt=opt,
        constraints=constraints,
        engrad=engrad
    )
    code = orca.run(str(orca_input.resolve()), job_path=output_dir)
    if code != 0:
        logger.error(f"ORCA calculation failed for {input_file}")
        raise RuntimeError(f"ORCA calculation failed for {input_file}")
    if convert2molden:
        orca.gwb2molden(
            str(inFile.stem)+".gwb",
            str(output_dir/f"{inFile.stem}.molden"),
            job_path=output_dir
        )
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--n_threads", type=int, default=48)
    parser.add_argument("--method", type=str, default="HF")
    parser.add_argument("--basis", type=str, default="6-31G*")
    parser.add_argument("--opt", action="store_true", default=False)
    parser.add_argument("--convert2molden", action="store_true", default=False)
    parser.add_argument("--aqueous", action="store_true")
    parser.add_argument("--constraint-file", type=str, default=None)
    # parser.add_argument()
    args = parser.parse_args()
    logger.info(f"Calculating energy by {__file__}")
 
    if args.constraint_file is not None:
        with open(args.constraint_file, "rb") as f:
            constraints = pickle.load(f)
    else:
        constraints = None

    calculate_energy_orca(
        args.input_file,
        args.output_dir,
        charge=args.charge,
        n_threads=args.n_threads,
        method=args.method,
        basis=args.basis,
        aqueous=args.aqueous,
        opt=args.opt,
        convert2molden=args.convert2molden,
        constraints=constraints,
        engrad=True
    )