from rdkit import Chem
from rdkit.Chem import rdFMCS
import numpy as np
import psi4
from typing import List
import os
from pathlib import Path
from parna.logger import getLogger
import subprocess

logger = getLogger(__name__)

def mk_psi4_ret(atom_names: List[str], coords: np.ndarray, charge: int, mult: int):

    ret = f"{charge} {mult}\n"
    for name, coord in zip(atom_names, coords):
        ret += f"{name} {coord[0]:>10.6f} {coord[1]:>10.6f} {coord[2]:>10.6f}\n"
    return ret


def mk_psi4_geometry(atom_names: List[str], coords: np.ndarray, charge: int, mult: int):

    ret = f"{charge} {mult}\n"
    for name, coord in zip(atom_names, coords):
        ret += f"{name} {coord[0]:>10.6f} {coord[1]:>10.6f} {coord[2]:>10.6f}\n"
    # print(ret)
    mol = psi4.geometry(ret)
    return mol


def calculate_energy_psi4(
        input_file,
        output_dir, 
        charge=0, 
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31G*",
        aqueous=False,
        area=0.3,
        script_run=False,
    ):
    areas = [0.2, 0.3, 0.6, 0.9, 1.2]
    if script_run:
        command = ["python", 
                   str((Path(__file__).parent/"psi4_calculate_energy.py").resolve()), 
                   input_file, str(output_dir), 
                   "--charge", str(charge),
                    "--memory", str(memory),
                    "--n_threads", str(n_threads),
                    "--method_basis", str(method_basis),
                   ]
        if aqueous:
            command.append("--aqueous")            
            command.append("--area")
            for area in areas:
                command_aq = command.copy()
                command_aq.append(str(area))
                logger.info(" ".join(command_aq))
                result = subprocess.run(command_aq)
                return_code = result.returncode

                if return_code == 0:
                    return result
                else:
                    logger.error(f"Failed to calculate energy with area={area}. Retrying to calculate with a different area...")
                    continue
            raise ValueError(f"Failed to calculate energy with all areas {areas}")   
    else:
        calculate_energy_function(
            input_file,
            output_dir,
            charge=charge,
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis,
            aqueous=aqueous,
        )

        


def calculate_energy_function(input_file,
                     output_dir, 
                     charge=0, 
                     memory="160 GB", 
                     n_threads=48, 
                     method_basis="HF/6-31G*",
                     aqueous=False,
    ):
    
    logger.info(f"calculating {method_basis} energy for " + str(input_file))
    output_dir = Path(output_dir)
    inFile = Path(input_file)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    psi4log = output_dir/f"{inFile.stem}.psi4.log"
    psi4tmp_dir = output_dir/"tmp"
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)
    psi4.core.set_output_file(str(psi4log), False)
    psi4_io = psi4.core.IOManager.shared_object()
    if not os.path.exists(psi4tmp_dir):
        os.makedirs(psi4tmp_dir, exist_ok=True)
    psi4_io.set_default_path(str(psi4tmp_dir.resolve()))

    if inFile.suffix == ".pdb":
        mymol = Chem.MolFromPDBFile(str(inFile), removeHs = False)
    elif inFile.suffix == ".xyz":
        mymol = Chem.MolFromXYZFile(str(inFile))
    else:
        raise ValueError("The input file is not supported")
    coords = mymol.GetConformer().GetPositions()
    atom_symbols = [atom.GetSymbol() for atom in mymol.GetAtoms()]
  
    # get heavy atoms, set frozen cartesian, atom id starts from 1
    psi4_mol = mk_psi4_geometry(
        atom_symbols, coords, charge, 1
    )
    logger.info("Running psi4...")
    if aqueous:
        logger.info("Running psi4 with PCM (water)...")
        psi4.set_options({
        'pcm': True,
        'pcm_scf_type': 'total',
        })

        pcm_string = """
        Units = Angstrom
        Medium {{
        SolverType = CPCM
        Solvent = Water
        }}

        Cavity {{
        RadiiSet = Bondi
        Type = GePol
        Scaling = True
        Area = {area}
        Mode = Implicit
        }}
        """
        # psi4.set_options({"pcm__input": pcm_string})
        areas = [0.3, 0.6, 0.9, 1.2]
        for area in areas:
            psi4.set_options({"pcm__input": pcm_string.format(area=area)})
            try:
                logger.info(f"Running psi4 with area={area}...")
                print(pcm_string.format(area=area))
                energy, wfn = psi4.energy(method_basis, molecule=psi4_mol, return_wfn=True)
                if energy is not None:
                    break
            except SystemExit as e:
                logger.error(f"Failed to calculate energy with area={area}. Retrying to calculate with a different area...")
                continue
            except:
                logger.error(f"Failed to calculate energy with area={area}. Retrying to calculate with a different area...")
                continue
        # energy, wfn = psi4.energy(method_basis, molecule=psi4_mol, return_wfn=True)
    else:
        energy, wfn = psi4.energy(method_basis, molecule=psi4_mol, return_wfn=True)
    psi4.driver.fchk(wfn, str(output_dir/f"{inFile.stem}.psi4.fchk"))
    return energy


psi4_template_head = """import os
memory {memory}
psi4_io.set_default_path('{tmp_dir}')
set_num_threads({n_threads})
set_output_file('{output_file}')
molecule meoh {{
"""
psi4_template_tail_gas="""}}
energy_scf, wfn = energy('{method_basis}', return_wfn=True)
"""
psi4_template_tail_solv="""}}
set {{
  scf_type pk
  pcm true
  pcm_scf_type total
}}
pcm = {{
  Units = Angstrom
  Medium {{
  SolverType = CPCM # IEFPCM| CPCM
  Solvent = Water
  }}
  Cavity {{
  RadiiSet = Bondi # Bondi| UFF| Allinger
  Type = GePol
  Scaling = True # radii for the spheres will be scaled by 1.2
  Area = {area}
  Mode = Implicit
  }}
}}
energy_scf, wfn = energy('{method_basis}', return_wfn=True)
fchk(wfn,'{wfn_file}')
"""

class Engine(object):
    def __init__(self, coordfile = None, input_file = None):
        if input_file is not None:
            self.read_input(input_file)

        if coordfile is not None:
            self.write_input(coordfile)

    def read_input(self, input_file):
        raise NotImplementedError

    def write_input(self, coordfile):
        raise NotImplementedError

    # def run(self, cmd, input_files, output_files, job_path = None):
    #     cwd = os.getcwd()
    #     if job_path is not None:
    #         os.chdir(job_path)
    #     subprocess.run([cmd, input_files, output_files], shell = True)
    #     os.chdir(cwd)


class EnginePsi4(Engine):
    def __init__(self, coordfile = None, input_file = None):
        if input_file is not None:
            self.read_input(input_file)

        if coordfile is not None:
            self.write_input(coordfile)

    def write_input(self, ret, tmp_dir=None, output_file='output.dat', wfn_file="wfn.fchk", method_basis='hf/6-31G*', memory="160 GB", solvent = None, area=0.3, filename = 'input.dat', job_path = None):
        # take coordinate file and make molecule object ??
        with open(filename, 'w') as outfile:
            outfile.write(psi4_template_head.format(memory=memory, n_threads=48, tmp_dir=tmp_dir, output_file=output_file))
            outfile.write(ret)
            if solvent is None:
                outfile.write(psi4_template_tail_gas.format(method_basis=method_basis, wfn_file=wfn_file)) #### set inside input file
            else:
                outfile.write(psi4_template_tail_solv.format(method_basis=method_basis, solvent=solvent, area=area, wfn_file=wfn_file))
        logger.info(f"Input file written to {filename}")

    def run(self, input_file, job_path = None):
        cwd = os.getcwd()
        if job_path is not None:
            os.chdir(job_path)
        code = os.system(f"psi4 {input_file}")
        os.chdir(cwd)
        return code

def calculate_energy_shell(input_file,
                     output_dir, 
                     charge=0, 
                     memory="160 GB", 
                     n_threads=48, 
                     method_basis="HF/6-31G*",
                     aqueous=False,
    ):
    logger.info(f"calculating {method_basis} energy for " + str(input_file))
    output_dir = Path(output_dir)
    inFile = Path(input_file)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    psi4log = output_dir/f"{inFile.stem}.psi4.log"
    psi4tmp_dir = output_dir/"tmp"
    if not os.path.exists(psi4tmp_dir):
        os.makedirs(psi4tmp_dir, exist_ok=True)
    if inFile.suffix == ".pdb":
        mymol = Chem.MolFromPDBFile(str(inFile), removeHs = False)
    elif inFile.suffix == ".xyz":
        mymol = Chem.MolFromXYZFile(str(inFile))
    else:
        raise ValueError("The input file is not supported")
    coords = mymol.GetConformer().GetPositions()
    atom_symbols = [atom.GetSymbol() for atom in mymol.GetAtoms()]
  
    # get heavy atoms, set frozen cartesian, atom id starts from 1
    psi4_ret = mk_psi4_ret(
        atom_symbols, coords, charge, 1
    )

    psi4_input = output_dir/f"{inFile.stem}.psi4.input"
    psi4_engine = EnginePsi4()
    # psi4_engine.write_input(psi4_ret, method_basis=method_basis, memory=memory, filename=str(psi4_input), solvent="Water")
    psi4_engine.write_input(
        psi4_ret,
        method_basis=method_basis,
        memory=memory,
        filename=str(psi4_input.resolve()),
        solvent="Water",
        area=0.3,
        tmp_dir=str(psi4tmp_dir.resolve()),
        output_file=str(psi4log.resolve()),
        wfn_file = str(output_dir/f"{inFile.stem}.psi4.fchk")
    )

    code = psi4_engine.run(str(psi4_input.resolve()), job_path=output_dir)
    print(code)


def read_energy_from_log(log_file: str, source='psi4') -> float:
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
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = float(line.split()[-1])
                return energy
        return None