from . import TorsionOptimizer, TorsionFragmentizer, DihedralScanner
from parna.utils import rd_load_file, map_atoms, SLURM_HEADER_CPU
from parna.constants import Hatree2kCalPerMol
from parna.qm.psi4_utils import read_energy_from_log
from parna.parm import parameterize, generate_frcmod
from parna.resp import RESP_fragment
import parmed as pmd
from parna.logger import getLogger
import openmm as mm
import openmm.app as app
from pathlib import Path
import numpy as np
import os
import yaml


logger = getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent / "template"


class MoleculeFactory:
    def __init__(
            self,
            mol_name: str = "mol",
            atom_type: str = "amber",
            threads: int = 48,
            memory: str = "160 GB",
            resp_method: str = "HF",
            resp_basis: str = "6-31G*",
            resp_n_conformers: int = 6,
            output_dir: str = None
        ):
        self.threads = threads
        self.memory = memory
        self.resp_method = resp_method
        self.resp_basis = resp_basis
        self.resp_n_conformers = resp_n_conformers
        self.mol_name = mol_name
        self.atom_type = atom_type
        self.output_dir = output_dir
    
    def charge_molecule(self, input_file, charge, output_dir):
        """
        Calculate RESP charges for a molecule.
        Input:
            input_file: str, `PDB` or `XYZ` file.
            charge: int, charge of the molecule
            output_dir: str, path to the output directory
        """
        RESP_fragment(
            str(input_file),
            charge,
            str(output_dir),
            self.mol_name,
            memory=self.memory, 
            n_threads=self.threads, 
            method_basis=f"{self.resp_method}/{self.resp_basis}",
            n_conformers=self.resp_n_conformers
        )
    
    def parameterize(self, pdb_file, output_dir):
        """
        Parameterize a molecule using antechamber and tleap.
        Input:
            pdb_file: str, `PDB` file
            output_dir: str, path to the output directory
        """
        output_dir = Path(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pdb_file = Path(pdb_file)
        tmp_mol2 = output_dir / "tmp.mol2"
        command_antechamber = [
            "antechamber",
            "-fi", "pdb",
            "-i", str(pdb_file),
            "-fo", "mol2",
            "-o", str(tmp_mol2),
            "-at", self.atom_type,
            "-pf", "y"
        ]
        os.system(" ".join(command_antechamber))
        generate_frcmod(
            input_file=str(tmp_mol2),
            output_file=str(output_dir/f"{pdb_file.stem}.frcmod"),
            sinitize=False
        )
        mol2_pmd = pmd.load_file(str(tmp_mol2))
        for atom in mol2_pmd.atoms:
            atom.charge = 0.0
        tmp_lib = (output_dir/f"{pdb_file.stem}.tmp.lib")
        tmp_pdb = (output_dir/f"{pdb_file.stem}.tmp.pdb")
        mol2_pmd.save(str(tmp_lib), overwrite=True)
        mol2_pmd.save(str(tmp_pdb), overwrite=True)
        parameterize(
            oligoFile=str(tmp_pdb),
            external_libs=str(tmp_lib), 
            additional_frcmods=str(output_dir/f"{pdb_file.stem}.frcmod"),
            output_dir=output_dir,
            prefix=self.mol_name,
            solvated=False,
            saveparm=True,
            check_atomtypes=False
        )
        for tmp_file in [tmp_mol2, tmp_lib, tmp_pdb]:
            if tmp_file.exists():
                os.remove(tmp_file)
    
    @staticmethod
    def calculate_mm_energy(parm, positions: np.ndarray):
        system = parm.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        context = mm.Context(system, mm.VerletIntegrator(0.001))
        context.setPositions(positions)
        energy = pmd.openmm.energy_decomposition(parm, context)
        return energy


class TorsionFactory(MoleculeFactory):
    def __init__(
            self, 
            order: int = 4, 
            panelty_weight: float = 0.1, 
            threshold: float = 0.01, 
            mol_name: str = "mol",
            atom_type: str = "amber",
            template: str = "A",
            scanning_steps: int = 24,
            method: str = "wB97X-V",
            basis: str = "def2-TZVP",
            threads: int = 48,
            memory: str = "160 GB",
            resp_method: str = "HF",
            resp_basis: str = "6-31G*",
            resp_n_conformers: int = 6,
            output_dir: str = None
        ):
        super().__init__(
            mol_name=mol_name,
            atom_type=atom_type,
            threads=threads, memory=memory,
            resp_method=resp_method, resp_basis=resp_basis,
            resp_n_conformers=resp_n_conformers,
            output_dir=output_dir
        )
        self.optimizer = TorsionOptimizer(order, panelty_weight, threshold)
        self.fragmentizer = TorsionFragmentizer()
        self.mol = None
        if template in ["A", "U", "G", "C"]:
            self.template_file = TEMPLATE_DIR / f"NA_{template}.pdb"
        elif template is not None:
            self.template_file = template
        else:
            self.template_file = None
        if self.template_file is not None:
            self.template_mol = rd_load_file(self.template_file)
        self.scanning_steps = scanning_steps
        self.start = -180
        self.end = 180
        self.basis = basis
        self.method = method
    
    def load_file(self, filename, charge=None):
        file_path = Path(filename)
        if file_path.suffix in [".pdb", '.xyz'] and charge is None:
            logger.error("Please provide charge for pdb or xyz file")
            raise ValueError("Please provide charge for pdb or xyz file")
        if file_path.suffix in [".pdb", '.xyz']:
            determine_bond_order = True
        else:
            determine_bond_order = False
        self.mol = rd_load_file(file_path, removeHs=False, sanitize=True, charge=charge, determine_bond_order=determine_bond_order)
    
    def fragmentize(self, output_dir):
        core_atoms = map_atoms(self.template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        n_atom_mod = self.mol.GetNumAtoms()
        unique_atoms = [i for i in range(n_atom_mod) if i not in [a[1] for a in core_atoms]]
        TF = TorsionFragmentizer(self.mol)
        TF.fragmentize()
        self.valid_fragments = {}
        for idx, frag in enumerate(TF.fragments):
            dq = frag.pivotal_dihedral_quartet[0]
            if (np.any([frag.fragment_parent_mapping[a] in unique_atoms for a in dq])):
                frag.save(str(output_dir/f"fragment{idx}"), format="pdb")
                logger.info(f"Fragment {idx} is saved to {str(output_dir/f'fragment{idx}')}")
                self.valid_fragments[idx] = frag
                
    def gen(self, submit=False, local=False):
        output_dir = Path(self.output_dir)
        logger.info(f" >>> Output directory: {output_dir} <<<")
        self.fragmentize(output_dir)
        self.all_conformer_files = {}
        self.slurm_files = []
        for vi, vfrag in self.valid_fragments.items():
            logger.info(f"Scanning dihedral {vfrag.pivotal_dihedral_quartet[0]}")
            dsc = DihedralScanner(
                input_file=output_dir/f"fragment{vi}/fragment.pdb", 
                dihedrals=[vfrag.pivotal_dihedral_quartet[0]],
                charge=0,
                workdir=output_dir/f"fragment{vi}",
            )

            dsc.run(
                start=self.start,
                end=self.end + (360/self.scanning_steps)*(self.scanning_steps-1),
                steps=self.scanning_steps
            )
            self.all_conformer_files[vi] = dsc.conformers[vfrag.pivotal_dihedral_quartet[0]]
        
            for idx, conf in enumerate(dsc.conformers[vfrag.pivotal_dihedral_quartet[0]]):
                slurm_file = output_dir/f"fragment{vi}/conf{idx}.slurm"
                if vfrag.charge != 0:
                    logger.warning(f"Fragment {vi} has charge {vfrag.charge}")
                    logger.warning(f"Please use appropriate QM methods for charged fragments")
                self.write_slurm(
                    slurm_filename=slurm_file,
                    input_file=str(conf),
                    output_dir=output_dir/f"fragment{vi}",
                    charge=vfrag.charge
                )
                if local:
                    cwd = Path.cwd().resolve()
                    os.chdir(output_dir)
                    os.system(f"bash {slurm_file}")
                    os.chdir(cwd)
                elif submit:
                    cwd = Path.cwd().resolve()
                    os.chdir(output_dir)
                    os.system(f"LLsub {slurm_file}")
                    os.chdir(cwd)
                self.slurm_files.append(slurm_file)
        
    def optimize(self):
        output_dir = Path(self.output_dir)
        self.conformer_data = {}
        self.parameter_set = {}
        for vi, vfrag in self.valid_fragments.items():
            self.conformer_data [vi] = {}
            logger.info(f"Optimizing fragment {vi}")
            logger.info(f"calculating atomic charges for fragment {vi}")
            frag_dir = Path(output_dir/f"fragment{vi}")
            self.charge_molecule(
                self.all_conformer_files[vi][0], 
                vfrag.charge, frag_dir
            )
            charged_mol2 = frag_dir/f"{self.all_conformer_files[vi][0].stem}.mol2"
            charged_pmd_mol = pmd.load_file(charged_mol2)
            logger.info("Parameterizing fragment")
            self.parameterize(str(output_dir/f"fragment{vi}/fragment.pdb"), frag_dir)
            parm7 = frag_dir/f"{self.mol_name}.parm7"
            parm_mol = pmd.load_file(parm7)
            for idx, atom in enumerate(parm_mol.atoms):
                atom.charge = charged_pmd_mol.atoms[idx].charge
            dih_idx = self.get_dihrdeal_term_by_quartet(parm_mol, vfrag.pivotal_dihedral_quartet[0])
            parm_mol.dihedrals[dih_idx].type.phi_k = 0.0
            logger.info(f'collecting QM and MM Energies for fragment {vi}')
            for conf in self.all_conformer_files[vi]:
                conf_mol = rd_load_file(str(conf))
                positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
                mm_energy_dict = self.calculate_mm_energy(parm_mol, positions)
                parm_mol.positions = positions
                log_file = conf.with_suffix(".psi4.HF.log")
                qm_energy = read_energy_from_log(log_file)
                self.conformer_data[vi][conf.stem] = {
                    "dihedral": parm_mol.dihedrals[dih_idx].measure(),
                    "mm_energy": mm_energy_dict["total"],
                    "qm_energy": qm_energy * Hatree2kCalPerMol
                    
                }

            dihedrals = np.array([self.conformer_data[vi][conf.stem]["dihedral"] for conf in self.all_conformer_files[vi]])
            mm_energy = np.array([self.conformer_data[vi][conf.stem]["mm_energy"] for conf in self.all_conformer_files[vi]])
            qm_energy = np.array([self.conformer_data[vi][conf.stem]["qm_energy"] for conf in self.all_conformer_files[vi]])
            mm_energy = mm_energy - mm_energy.min()
            qm_energy = qm_energy - qm_energy.min()
            self.optimizer.infer_parameters(
                dihedrals=dihedrals,
                energy_mm=mm_energy,
                energy_qm=qm_energy
            )
            self.parameter_set[vi] = self.optimizer.get_parameters()
            parent_dihedral_index = [vfrag.fragment_parent_mapping[a] for a in vfrag.pivotal_dihedral_quartet[0]]
            self.parameter_set[vi]["dihedal"] = parent_dihedral_index
            
        self.save_to_yaml(self.conformer_data, output_dir/"conformer_data.yaml")
        self.save_to_yaml(self.parameter_set, output_dir/"parameter_set.yaml")
            
    
    @staticmethod    
    def save_to_yaml(dict_data, yaml_file):
        with open(yaml_file, "w") as f:
            yaml.dump(dict_data, f)
        
    
    def write_slurm(self, slurm_filename, input_file, output_dir, charge):
        slurm_content = SLURM_HEADER_CPU.copy()
        psi4_python_exec = Path(__file__).parent.parent / "qm" / "psi4_calculate_energy.py"
        slurm_content += f"python" \
                            f" {str(psi4_python_exec.resolve())}" \
                            f" {str(input_file)}" \
                            f" {str(output_dir)}" \
                            f" --charge {charge}" \
                            f" --memory {self.memory}" \
                            f" --n_threads {self.threads}" \
                            f" --method_basis {self.method}/{self.basis}" \
                            f"\n"
        with open(slurm_filename, "w") as f:
            f.write(slurm_content)
    
    def get_dihrdeal_term_by_quartet(self, parm_mol, atom_index_list):
        atom_index_list = np.array(sorted(atom_index_list))
        for idx, dihedral in enumerate(parm_mol.dihedrals):
            _atom_index_list = np.array(sorted([dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]))
            if np.all(_atom_index_list == atom_index_list):
                return idx
    
    
        
    
       



