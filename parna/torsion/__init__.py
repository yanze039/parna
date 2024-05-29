from .fragment import TorsionFragmentizer
from .optim import TorsionOptimizer
from .module import TorsionFactory

from typing import List, Tuple
import os
from pathlib import Path
import shutil
from parna.qm.xtb import XTB
from parna.utils import split_xyz_file


class DihedralScanner:
    def __init__(
                self, 
                input_file: str, 
                dihedrals: List[Tuple[int, int, int, int]],
                engine="xtb",
                charge=0,
                workdir="xtb",
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
            self.engine = XTB(force_constant=0.05)
        else:
            raise ValueError("Unsupported engine")
        self.conformers = {}
        self.conformer_prefix = "conformer"
    
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
    
    def add_constraint(self, dihedral: Tuple[int, int, int, int], angle: float):
        """
        Add a constraint to the dihedral angle.
        Input:
            dihedral: a tuple of four atom indices
            angle: the angle to constrain the dihedral angle to.
        """
        self.engine.add_constraint(
            constraint_type="dihedral",
            atoms = dihedral, 
            at = angle,
        )
    
    def clear(self):
        """
        Clear all constraints.
        """
        self.engine.clear()
        self.conformers = {}
        
    def _scan(self, dihedral: Tuple[int, int, int, int], start: float, end: float, steps: int):
        """
        Scan a dihedral angle in a molecule.
        """
        
        self.engine.set_scan(
            scan_type="dihedral",
            atoms=dihedral,
            start=start,
            end=end,
            steps=steps
        )
        self.engine.run(
            coord=self.input_file,
            charge=self.charge,
            workdir=self.workdir,
            inp_name="xtb_scan.inp"
        )
        _tmp_log = self.workdir/"xtbscan.log"
        split_xyz_file(_tmp_log, output_folder=self.workdir, output_prefix=self.conformer_prefix)
        for i in range(steps):
            self.add_conformer(dihedral, (self.workdir/f"{self.conformer_prefix}_{i}.xyz").resolve())
    
    def run(self, start: float, end: float, steps: int):
        """
        Run the scan.
        """
        for dihedral in self.dihedrals:
            self._scan(dihedral, start, end, steps)
    
    def get_results(self):
        """
        Get the results of the scan.
        """
        return self.engine.get_results()
        
