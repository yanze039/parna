
from typing import List, Tuple
import os
from pathlib import Path
import shutil
from parna.qm.xtb_utils import XTB as EngineXTB
from parna.qm.orca_utils import EngineORCA
from parna.utils import split_xyz_file
from parna.logger import getLogger
import rdkit.Chem as Chem
import numpy as np
from ase import Atoms
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator, all_properties
from ase.io import read, write
# ase.io.xyz.write_xyz
from ase.io.xyz import write_xyz
import openmm as mm
import openmm.app as app
import openmm.unit as unit

logger = getLogger(__name__)


class DihedralScanner:
    def __init__(
                self, 
                input_file: str, 
                dihedrals: List[Tuple[int, int, int, int]],
                engine="xtb",
                charge=0,
                workdir="xtb",
                conformer_prefix="conformer",
                constraints=None,
                force_constant=1.0,
                warming_constraints=True
        ):
        self.input_file = input_file
        self.dihedrals = dihedrals
        self.engine = engine
        self.charge = charge
        self.workdir = Path(workdir)
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        if engine == "xtb":
            if not shutil.which("xtb"):
                raise FileNotFoundError("xtb is not found in the PATH. Please install xtb and add it to the PATH.")
            self.engine = EngineXTB(force_constant=force_constant)
        else:
            raise ValueError("Unsupported engine")
        self.conformers = {}
        self.conformer_prefix = conformer_prefix
        self.constraints = constraints
        self.warming_constraints = warming_constraints
        self.force_constant = force_constant
        
    
    def add_conformer(self, dihedral: Tuple[int, int, int, int], conformer):
        """
        Add a conformer to the scan.
        Input:
            dihedral: a tuple of four atom indices
            conformer: a rdkit.Chem.Conformer object
        """
        if dihedral in self.conformers:
            key = dihedral
        elif dihedral[::-1] in self.conformers:
            key = dihedral[::-1]
        else:
            key = dihedral
            self.conformers[key] = []
        self.conformers[key].append(conformer)
    
    def get_conformers(self, dihedral: Tuple[int, int, int, int]):
        """
        Get the conformers for a dihedral angle.
        """
        if dihedral in self.conformers:
            return self.conformers[dihedral]
        elif dihedral[::-1] in self.conformers:
            return self.conformers[dihedral[::-1]]
        else:
            return []
    
    def add_constraint(self, dihedral: Tuple[int, int, int, int], angle: float, foece_constant=1.0):
        """
        Add a constraint to the dihedral angle.
        Input:
            dihedral: a tuple of four atom indices (0-indexed)
            angle: the angle to constrain the dihedral angle to.
        """
        self.engine.add_constraint(
            constraint_type="dihedral",
            atoms = dihedral, 
            at = angle,
            force_constant=foece_constant
        )
    
    def clear_constraints(self):
        """
        Clear all constraints.
        """
        self.engine.clear_constraints()
    
    def clear(self):
        """
        Clear all constraints.
        """
        self.engine.clear()
        self.conformers = {}
        
    def _scan(self, dihedral: Tuple[int, int, int, int], 
              start: float, end: float, steps: int, 
              overwrite:bool=False):
        """
        Scan a dihedral angle in a molecule.
        """
        for i in range(steps):
            self.add_conformer(dihedral, (self.workdir/f"{self.conformer_prefix}_{i}.xyz").resolve())
            # just to record the conformers
        
        if (not np.all([os.path.exists(self.workdir/f"{self.conformer_prefix}_{i}.xyz") for i in range(steps)])) or overwrite:
            self.engine.set_scan(
                scan_type="dihedral",
                atoms=dihedral,
                start=start,
                end=end,
                steps=steps
            )
            if self.constraints is not None and len(self.constraints) > 0:
                for c in self.constraints:
                    self.add_constraint(c[0], c[1])
            self.engine.run(
                coord=self.input_file,
                charge=self.charge,
                workdir=self.workdir,
                inp_name="xtb_scan.inp"
            )
            _tmp_log = self.workdir/"xtbscan.log"
            split_xyz_file(_tmp_log, output_folder=self.workdir, output_prefix=self.conformer_prefix)
        else:
            logger.info("Conformers already exist. Skipping the scan.")
    
    
    def _scan_on_grids(self, dihedral, grids, concurrent_constraints = None,
                       overwrite:bool=False):
        """
        Scan the dihedral angle on a grid.
        concurrent_constraints: a list of constraints for each grid point.
        """
                
        for i in range(len(grids)):
            self.clear_constraints()
            self.add_conformer(dihedral, (self.workdir/f"{self.conformer_prefix}_{i}.xyz").resolve())
            if (not os.path.exists(self.workdir/f"{self.conformer_prefix}_{i}.xyz")) or overwrite:
                if self.constraints is not None and len(self.constraints) > 0:
                    for c in self.constraints:
                        self.add_constraint(c[0], c[1], foece_constant=self.force_constant/20.)
                if concurrent_constraints is not None and len(concurrent_constraints) > 0:
                    for c in concurrent_constraints[i]:
                        print(c)
                        self.add_constraint(c[0], c[1], foece_constant=self.force_constant/20.)
                self.add_constraint(dihedral, grids[i], foece_constant=self.force_constant/20.)
                self.engine.run(
                    coord=self.input_file,
                    charge=self.charge,
                    workdir=self.workdir,
                    inp_name="xtb_scan.inp"
                )
                self.clear_constraints()
                if self.constraints is not None and len(self.constraints) > 0:
                    for c in self.constraints:
                        self.add_constraint(c[0], c[1], foece_constant=self.force_constant)
                if concurrent_constraints is not None and len(concurrent_constraints) > 0:
                    for c in concurrent_constraints[i]:
                        self.add_constraint(c[0], c[1], foece_constant=self.force_constant)
                self.add_constraint(dihedral, grids[i], foece_constant=self.force_constant)
                shutil.copy(self.workdir/"xtbopt.pdb", self.workdir/"xtbopt.tmp.pdb")
                self.engine.run(
                    coord=self.workdir/"xtbopt.tmp.pdb",
                    charge=self.charge,
                    workdir=self.workdir,
                    inp_name="xtb_scan.inp"
                )
                tmp_mol = Chem.MolFromPDBFile(str(self.workdir/"xtbopt.pdb"), removeHs=False)
                Chem.MolToXYZFile(tmp_mol, str(self.workdir/f"{self.conformer_prefix}_{i}.xyz"))
                os.remove(self.workdir/"xtbopt.pdb")
                os.remove(self.workdir/"xtbopt.tmp.pdb")
        else:
            logger.info("Conformers already exist. Skipping the scan.")
        
    
    def run(self, start: float, end: float, steps: int, overwrite:bool=False):
        """
        Run the scan.
        """
        for dihedral in self.dihedrals:
            self._scan(dihedral, start, end, steps, overwrite=overwrite)
    
    def run_on_grids(self, grids: List[float], overwrite:bool=False, concurrent_constraints=None):
        """
        Run the scan on a grid.
        """
        for dihedral in self.dihedrals:
            self._scan_on_grids(dihedral, grids, overwrite=overwrite, concurrent_constraints=concurrent_constraints)
    
    def get_results(self):
        """
        Get the results of the scan.
        """
        return self.engine.get_results()
        

class ConformerOptimizer:
    def __init__(
                self, 
                input_file: str, 
                engine="xtb",
                charge=0,
                workdir="xtb",
                conformer_prefix="conformer",
                constraints=None,
                force_constant=1.0,
                warming_constraints=True
        ):
        self.input_file = Path(input_file)
        
        if self.input_file.suffix == ".pdb":
            rdmol = Chem.MolFromPDBFile(str(self.input_file), removeHs=False)
            Chem.MolToXYZFile(rdmol, str(self.input_file.with_suffix(".conformer_opt.xyz")))
            self.input_file = self.input_file.with_suffix(".conformer_opt.xyz")
        
        self.engine = engine
        self.charge = charge
        self.workdir = Path(workdir).resolve()
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        if engine.lower() == "xtb":
            if not shutil.which("xtb"):
                raise FileNotFoundError("xtb is not found in the PATH. Please install xtb and add it to the PATH.")
            self.engine_name = "xtb"
            self.engine = EngineXTB(force_constant=force_constant)
        elif engine.lower() == "orca":
            self.engine = EngineORCA()
            self.engine_name = "orca"
        else:
            raise ValueError("Unsupported engine")
        self.conformers = {}
        self.conformer_prefix = conformer_prefix
        self.constraints = constraints
        self.warming_constraints = warming_constraints
        self.force_constant = force_constant
    
    def add_constraint(self, dihedral: Tuple[int, int, int, int], angle: float, foece_constant=1.0):
        """
        Add a constraint to the dihedral angle.
        Input:
            dihedral: a tuple of four atom indices (0-indexed)
            angle: the angle to constrain the dihedral angle to.
        """
        self.engine.add_constraint(
            constraint_type="dihedral",
            atoms = dihedral, 
            at = angle,
            force_constant=foece_constant
        )
    
    def clear_constraints(self):
        """
        Clear all constraints.
        """
        self.engine.clear_constraints()
    
    def clear(self):
        """
        Clear all constraints.
        """
        self.engine.clear()
        self.conformers = {}
    
    def set_sampling(self):
        self.engine.set_sampling()

    def run_xtb(self, overwrite=True, clean_tmp=True, sampling=True):
        """
        Scan the dihedral angle on a grid.
        concurrent_constraints: a list of constraints for each grid point.
        """
        assert self.engine_name == "xtb"
        self.output_file = self.workdir/f"{self.conformer_prefix}_opt.xyz"
        if not overwrite and os.path.exists(self.workdir/f"{self.conformer_prefix}_opt.xyz"):
            logger.info("Conformer already exists. Skipping the optimization.")
            return
        
        tmp_workdir = self.workdir/f"xtb_tmp_{self.conformer_prefix}"
        # if not os.path.exists(tmp_workdir):
        tmp_workdir.mkdir(exist_ok=True)
        
        if self.warming_constraints:
            self.clear_constraints()
            if self.constraints is not None and len(self.constraints) > 0:
                for c in self.constraints:
                    self.add_constraint(c[0], c[1], foece_constant=self.force_constant/20.)
            self.engine.run(
                coord=str(self.input_file),
                charge=self.charge,
                workdir=tmp_workdir,
                inp_name=f"{self.conformer_prefix}.opt1.inp",
                solution=None,
                sampling=sampling,
                opt=False,
                gfnff=True,
            )
            output_traj = tmp_workdir/f"xtb.trj"
            split_xyz_file(input_file_path=output_traj, 
                           output_folder=tmp_workdir, 
                           every_frame=1,
                           output_prefix=self.conformer_prefix+"_sampled")
            traj_list = sorted(list(tmp_workdir.glob(f"{self.conformer_prefix}_sampled_*.xyz")))
            if len(traj_list) < 2:
                raise ValueError("Sampling failed.")
            last_xyz_file = traj_list[-1]
            shutil.copy(last_xyz_file, tmp_workdir/f"xtbopt.tmp{self.input_file.suffix}")
            for traj_frame in traj_list:
                os.remove(traj_frame)
            # os.remove(tmp_workdir/f"xtbrestart")
            os.remove(tmp_workdir/f"mdrestart")
            os.remove(tmp_workdir/f"xtbmdok")
            os.remove(tmp_workdir/f"xtb.trj")
            
        self.clear_constraints()
        if self.constraints is not None and len(self.constraints) > 0:
            for c in self.constraints:
                self.add_constraint(c[0], c[1], foece_constant=self.force_constant/5.)
        if not self.warming_constraints:
            tmp_file = self.input_file
        else:
            tmp_file = tmp_workdir/f"xtbopt.tmp{self.input_file.suffix}"
        if sampling:
            self.engine.run(
                coord=str(tmp_file),
                charge=self.charge,
                workdir=tmp_workdir,
                inp_name=f"{self.conformer_prefix}.md.inp",
                opt=False,
                sampling=sampling,
                solution=None
            )
            output_traj = tmp_workdir/f"xtb.trj"
            split_xyz_file(input_file_path=output_traj, 
                           output_folder=tmp_workdir, 
                           every_frame=1,
                           output_prefix=self.conformer_prefix+"_sampled")
            traj_list = sorted(list(tmp_workdir.glob(f"{self.conformer_prefix}_sampled_*.xyz")))
            # read energy in xyz file 
            energies = []
            for traj_frame in traj_list:
                with open(traj_frame, "r") as f:
                    lines = f.readlines()
                    energies.append(float(lines[1].strip().split()[1]))
            min_energy_idx = np.argmin(energies)
            if len(traj_list) < 2:
                raise ValueError("Sampling failed.")
            logger.info(f"Minimum energy frame: {traj_list[min_energy_idx]}")
            last_xyz_file = traj_list[min_energy_idx]
            # last_xyz_file = traj_list[-1]
            tmp_file = last_xyz_file
        self.clear_constraints()
        if self.constraints is not None and len(self.constraints) > 0:
            for c in self.constraints:
                self.add_constraint(c[0], c[1], foece_constant=self.force_constant)
        self.engine.run(
            coord=str(tmp_file),
            charge=self.charge,
            workdir=tmp_workdir,
            inp_name=f"{self.conformer_prefix}.opt2.inp",
            opt=True,
            sampling=False
        )
        if self.input_file.suffix == ".xyz":
            os.rename(tmp_workdir/"xtbopt.xyz", self.workdir/f"{self.conformer_prefix}_opt.xyz")
        elif self.input_file.suffix == ".pdb":
            tmp_mol = Chem.MolFromPDBFile(str(tmp_workdir/"xtbopt.pdb"), removeHs=False)
            Chem.MolToXYZFile(tmp_mol, str(self.workdir/f"{self.conformer_prefix}_opt.xyz"))
        if clean_tmp:
            if os.path.exists(tmp_workdir/f"xtbopt{self.input_file.suffix}"):
                os.remove(tmp_workdir/f"xtbopt{self.input_file.suffix}")
            if self.warming_constraints:
                os.remove(tmp_workdir/f"xtbopt.tmp{self.input_file.suffix}")
            if os.path.exists(tmp_workdir):
                shutil.rmtree(tmp_workdir)
        
        
    def run_orca(self, basis="", method="XTB2", convergence="normal", solvent="water", n_proc=1, overwrite=True, clean_tmp=True):
        self.output_file = self.workdir/f"{self.conformer_prefix}_opt.xyz"
        if not overwrite and os.path.exists(self.workdir/f"{self.conformer_prefix}_opt.xyz"):
            logger.info("Conformer already exists. Skipping the optimization.")
            return
        scrach_dir = self.workdir/f"orca_tmp_{self.conformer_prefix}"
        scrach_dir.mkdir(exist_ok=True)
        self.engine.write_input(
            basis,
            method,
            str(self.input_file),
            opt=True,
            solvent=solvent,
            n_proc=n_proc,
            charge=self.charge,
            orca_input_file=str(scrach_dir/f"{self.conformer_prefix}.inp"),
            constraints=self.constraints,
            convergence=convergence
        )
        exit_code = self.engine.run(str(scrach_dir/f"{self.conformer_prefix}.inp"), job_path=scrach_dir)
        
        if exit_code != 0:
            logger.error(f"ORCA calculation failed for {self.input_file}")
            raise RuntimeError(f"ORCA calculation failed for {self.input_file}")
        
        shutil.copy(scrach_dir/f"{self.conformer_prefix}.xyz", self.output_file)
        # exit(0)
        if clean_tmp:
            shutil.rmtree(scrach_dir)
    
    def run(self, method=None, basis=None, convergence="normal", 
            solvent=None, n_proc=1, overwrite=True, clean_tmp=True, sampling=False):
        if self.engine_name == "xtb":
            self.run_xtb(overwrite=overwrite, clean_tmp=clean_tmp, sampling=sampling)
        elif self.engine_name == "orca":
            self.run_orca(basis, method, convergence=convergence, solvent=solvent, n_proc=n_proc, overwrite=overwrite, clean_tmp=clean_tmp)
        else:
            raise ValueError("Unsupported engine")
    
        
class ConformerOptimizerCMAP:
    def __init__(
                self, 
                input_file: str, 
                engine="xtb",
                charge=0,
                workdir="xtb",
                conformer_prefix="conformer",
                constraints=None,
                force_constant=1.0,
                warming_constraints=True
        ):
        self.input_file = Path(input_file)
        self.engine = engine
        self.charge = charge
        self.workdir = Path(workdir)
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        if engine == "xtb":
            if not shutil.which("xtb"):
                raise FileNotFoundError("xtb is not found in the PATH. Please install xtb and add it to the PATH.")
            self.engine = EngineXTB(force_constant=force_constant)
        else:
            raise ValueError("Unsupported engine")
        self.conformers = {}
        self.conformer_prefix = conformer_prefix
        self.constraints = constraints
        self.warming_constraints = warming_constraints
        self.force_constant = force_constant
    
    def add_constraint(self, dihedral: Tuple[int, int, int, int], angle: float, foece_constant=1.0):
        """
        Add a constraint to the dihedral angle.
        Input:
            dihedral: a tuple of four atom indices (0-indexed)
            angle: the angle to constrain the dihedral angle to.
        """
        self.engine.add_constraint(
            constraint_type="dihedral",
            atoms = dihedral, 
            at = angle,
            force_constant=foece_constant
        )
    
    def clear_constraints(self):
        """
        Clear all constraints.
        """
        self.engine.clear_constraints()
    
    def clear(self):
        """
        Clear all constraints.
        """
        self.engine.clear()
        self.conformers = {}

    def run(self, relaxing_dihedrals: List[Tuple[int, int, int, int]]):
        """
        Scan the dihedral angle on a grid.
        concurrent_constraints: a list of constraints for each grid point.
        """
        self.clear_constraints()
        if self.constraints is not None and len(self.constraints) > 0:
            for c in self.constraints:
                self.add_constraint(c[0], c[1], foece_constant=self.force_constant/20.)
        self.engine.run(
            coord=self.input_file,
            charge=self.charge,
            workdir=self.workdir,
            inp_name="xtb_opt.inp"
        )
        shutil.copy(self.workdir/f"xtbopt{self.input_file.suffix}", self.workdir/f"xtbopt.tmp{self.input_file.suffix}")
        self.clear_constraints()
        if self.input_file.suffix == ".xyz":
            mol = Chem.MolFromXYZFile(str(self.workdir/f"xtbopt.tmp{self.input_file.suffix}"))
        elif self.input_file.suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(self.workdir/"xtbopt.tmp.pdb"), removeHs=False)
        else:
            raise ValueError("Unsupported file format")
        new_constraints = []
        for dihedral in relaxing_dihedrals:
            at = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *dihedral)
            # get the nearest degree on the grids, resolution=360/24
            at = round(at/15)*15
            new_constraints.append([dihedral, at])
        for nc in new_constraints:
            self.add_constraint(nc[0], nc[1], foece_constant=self.force_constant)
        if self.constraints is not None and len(self.constraints) > 0:
            for c in self.constraints:
                self.add_constraint(c[0], c[1], foece_constant=self.force_constant)
        
        
        self.engine.run(
            coord=str(self.workdir/f"xtbopt.tmp{self.input_file.suffix}"),
            charge=self.charge,
            workdir=self.workdir,
            inp_name="xtb_scan.inp"
        )
        if self.input_file.suffix == ".xyz":
            os.rename(self.workdir/"xtbopt.xyz", self.workdir/f"{self.conformer_prefix}_opt.xyz")
        elif self.input_file.suffix == ".pdb":
            tmp_mol = Chem.MolFromPDBFile(str(self.workdir/"xtbopt.pdb"), removeHs=False)
            Chem.MolToXYZFile(tmp_mol, str(self.workdir/f"{self.conformer_prefix}_opt.xyz"))
            os.remove(self.workdir/f"xtbopt{self.input_file.suffix}")
        else:
            raise ValueError("Unsupported file format")
        os.remove(self.workdir/f"xtbopt.tmp{self.input_file.suffix}")
        return self.constraints + new_constraints
    
        
class OpenMMCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, parm_mol, device="CUDA", properties={}):
        super().__init__()
        temperature = 300.
        system = parm_mol.createSystem(
                nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                implicitSolvent=app.HCT) # corresponding to igb=1 in Amber)
        for f in system.getForces():
            f.setForceGroup(0)
        integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, 1.0*unit.femtosecond)
        integrator.setRandomNumberSeed(1106)
        minimization_platform = mm.Platform.getPlatformByName(device)
        simulation = app.Simulation(parm_mol.topology, system, integrator, minimization_platform, properties)
        self.context = simulation.context
    
    def calculate(self, atoms, properties=all_properties, system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions() * unit.angstrom  # Convert positions to OpenMM units        
        self.context.setPositions(positions)
        state = self.context.getState(getEnergy=True, getForces=True, groups={0})

        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 96.485
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.angstrom) / 96.485

        self.results['energy'] = energy
        self.results['forces'] = np.array(forces)


def calculate_relaxed_energy(atoms, calculator, fmax=0.01, epsilon=1e-5, steps=10000, logfile="-", dihedral_constraints=None, save=None):
    # atoms = read(coords_files)
    atoms.calc = calculator
    
    if dihedral_constraints is not None:
        c = FixInternals(dihedrals_deg=dihedral_constraints, epsilon=epsilon)
        atoms.set_constraint(c)
    dyn = BFGS(atoms, logfile=logfile)
    dyn.run(fmax=fmax, steps=steps)
    # Save in XYZ format
    if save is not None:
        with open(save, "w") as f:
            write_xyz(f, [atoms])
    return atoms.get_potential_energy()
    