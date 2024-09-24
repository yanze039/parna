
from typing import List, Tuple
import os
from pathlib import Path
import shutil
from parna.qm.xtb import XTB
from parna.utils import split_xyz_file
from parna.logger import getLogger
import rdkit.Chem as Chem
import numpy as np


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
            self.engine = XTB(force_constant=force_constant)
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
        
