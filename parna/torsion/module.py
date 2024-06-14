from .optim import TorsionOptimizer
from .fragment import TorsionFragmentizer, C5Fragmentizer
from .conf import DihedralScanner
from parna.utils import rd_load_file, map_atoms, SLURM_HEADER_CPU, atomName_to_index
from parna.constant import Hatree2kCalPerMol
from parna.qm.psi4_utils import read_energy_from_log
from parna.qm.orca_utils import read_energy_from_txt
from parna.resp import RESP
from parna.molops import MoleculeFactory
import parmed as pmd
from parna.logger import getLogger
from pathlib import Path
import numpy as np
import rdkit.Chem as Chem
import os
import yaml


logger = getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent / "template"


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
            output_dir: str = None,
            cap_methylation: bool = True,
            aqueous: bool = False,
            fix_phase: bool = True
        ):
        super().__init__(
            mol_name=mol_name,
            atom_type=atom_type,
            threads=threads, memory=memory,
            resp_method=resp_method, resp_basis=resp_basis,
            resp_n_conformers=resp_n_conformers,
            output_dir=output_dir
        )
        self.order = order
        self.panelty_weight = panelty_weight
        self.threshold = threshold
        self.cap_methylation = cap_methylation
        self.aqueous = aqueous
        self.fix_phase = fix_phase
        
        self.mol = None
        if template in ["A", "U", "G", "C"]:
            self.template_file = TEMPLATE_DIR / f"NA_{template}.pdb"
        elif template is not None:
            self.template_file = template
        else:
            self.template_file = None
        if self.template_file is not None:
            self.template_mol = rd_load_file(self.template_file)
        
        self.sugar_template_file = str(TEMPLATE_DIR/"sugar_template.pdb")
        self.sugar_template_mol = Chem.MolFromPDBFile(str(self.sugar_template_file), removeHs=False)
        
        self.scanning_steps = scanning_steps
        self.start = -180
        self.end = 180
        self.basis = basis
        self.method = method
    
    def load_file(self, filename, charge=None):
        """
        Load a molecule from a file.
        Input: 
            filename: str, path to the file. Supported formats are `PDB`, `XYZ`, `SDF`
                    if `PDB` or `XYZ` file is provided, charge must be provided.
            charge: int, charge of the molecule
        """
        file_path = Path(filename)
        if file_path.suffix in [".pdb", '.xyz'] and charge is None:
            logger.error("Please provide charge for pdb or xyz file")
            raise ValueError("Please provide charge for pdb or xyz file")
        if file_path.suffix in [".pdb", '.xyz']:
            determine_bond_order = True
        else:
            determine_bond_order = False
        self.mol = rd_load_file(file_path, removeHs=False, sanitize=True, charge=charge, determine_bond_order=determine_bond_order)
        
    def match_template(self):
        self.core_atoms = map_atoms(self.template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        self.sugar_fragment_mapping = map_atoms(self.sugar_template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        sugar_name_to_index = atomName_to_index(self.sugar_template_mol)
        C1_index = sugar_name_to_index["C1'"]
        mapping_dict = {}
        for atom in self.sugar_fragment_mapping:
            mapping_dict[atom[0]] = atom[1]
        self.sugar_fragment_mapping = mapping_dict
        self.C1_index_parent = self.sugar_fragment_mapping[C1_index]
    
    def fragmentize(self, output_dir):
        self.match_template()
        n_atom_mod = self.mol.GetNumAtoms()
        unique_atoms = [i for i in range(n_atom_mod) if i not in [a[1] for a in self.core_atoms]]
        TF = TorsionFragmentizer(self.mol, cap_methylation=self.cap_methylation)
        TF.fragmentize()
        self.valid_fragments = {}
        for idx, frag in enumerate(TF.fragments):
            dq = frag.pivotal_dihedral_quartet[0]
            if (np.any([frag.fragment_parent_mapping[a] in unique_atoms for a in dq])) or \
                        self.C1_index_parent in [frag.fragment_parent_mapping[a] for a in frag.rotatable_bond]:
                frag.save(str(output_dir/f"fragment{idx}"), format="pdb")
                logger.info(f"Rotatable bond: {frag.rotatable_bond} is valid for fragment {idx}. Parent image: {[frag.fragment_parent_mapping[a] for a in frag.rotatable_bond]}")
                logger.info(f"Fragment {idx} is saved to {str(output_dir/f'fragment{idx}')}")
                self.valid_fragments[idx] = frag
        logger.debug(f"Valid fragments: {self.valid_fragments}")
                
    def gen(self, submit=False, local=False, overwrite=False):
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
                conformer_prefix=f"frag_conformer",
            ) 
            dsc.run(
                start=self.start,
                end=self.end + (360/self.scanning_steps)*(self.scanning_steps-1),
                steps=self.scanning_steps,
                overwrite=overwrite
            )
            self.all_conformer_files[vi] = dsc.conformers[vfrag.pivotal_dihedral_quartet[0]]
        
            for idx, conf in enumerate(dsc.conformers[vfrag.pivotal_dihedral_quartet[0]]):
                slurm_file = (output_dir/f"fragment{vi}/conf{idx}.slurm").resolve()
                if vfrag.charge != 0:
                    logger.warning(f"Fragment {vi} has charge {vfrag.charge}")
                    logger.warning(f"Please use appropriate QM methods for charged fragments")
                if self.aqueous:
                    self.write_script_aq(
                        slurm_filename=slurm_file,
                        input_file=str(conf),
                        output_dir=(output_dir/f"fragment{vi}").resolve(),
                        charge=vfrag.charge,
                        local=local
                    )
                else:
                    self.write_script(
                        slurm_filename=slurm_file,
                        input_file=str(conf),
                        output_dir=(output_dir/f"fragment{vi}").resolve(),
                        charge=vfrag.charge,
                        local=local
                    )
                self.slurm_files.append(slurm_file)
                
                if not overwrite and ((os.path.exists(conf.with_suffix(".psi4.log")) and not self.aqueous) or (os.path.exists(conf.with_suffix(".molden")) and self.aqueous)):
                    continue
                
                if local:
                    cwd = Path.cwd().resolve()
                    os.chdir(output_dir)
                    logger.info(f"Running {slurm_file}")
                    os.system(f"bash {slurm_file}")
                    os.chdir(cwd)
                elif submit:
                    cwd = Path.cwd().resolve()
                    os.chdir(output_dir)
                    logger.info(f"Submiting {slurm_file}")
                    os.system(f"LLsub {slurm_file}")
                    os.chdir(cwd)
                
        
    def optimize(self, overwrite=False):
        """
        Optimize the torsion parameters for the molecule.
        Input:
            overwrite: bool, if True, overwrite the existing files.
        Output:
            Save the conformer data and parameter set to the output directory.
        
        Description:
            1. Calculate atomic charges for each fragment.
            2. Parameterize each fragment.
            3. Collect QM and MM energies for each conformer.
            4. Optimize the torsion parameters.
        """
        output_dir = Path(self.output_dir)
        self.conformer_data = {}
        self.parameter_set = {}
        for vi, vfrag in self.valid_fragments.items():
            self.conformer_data [vi] = {}
            logger.info(f"Optimizing fragment {vi}")
            logger.info(f"calculating atomic charges for fragment {vi}")
            frag_dir = Path(output_dir/f"fragment{vi}")
            # get atomic charges
            charged_mol2 = frag_dir/f"{self.all_conformer_files[vi][0].stem}.mol2"
            if overwrite or (not os.path.exists(charged_mol2)):
                self.charge_molecule(
                    self.all_conformer_files[vi][0], 
                    vfrag.charge, frag_dir
                )
            else:
                logger.info(f"File exised, using existing charge file {charged_mol2}.")
            charged_pmd_mol = pmd.load_file(str(charged_mol2))
            logger.info("Parameterizing fragment")
            parm7 = frag_dir/f"{self.mol_name}_frag{vi}.parm7"
            if overwrite or (not os.path.exists(parm7)):
                if self.aqueous:
                    addons = ["set default PBRadii mbondi"]
                else:
                    addons = []
                self.parameterize(str(output_dir/f"fragment{vi}/fragment.pdb"), frag_dir, prefix=f"{self.mol_name}_frag{vi}", addons=addons)
            else:
                logger.info(f"File exised, using existing parameter file {parm7}.")
            parm_mol = pmd.load_file(str(parm7))
            for idx, atom in enumerate(parm_mol.atoms):
                atom.charge = charged_pmd_mol.atoms[idx].charge
            idx_list = self.get_dihrdeal_terms_by_quartet(parm_mol, vfrag.pivotal_dihedral_quartet[0])
            for dih_idx in idx_list:
                parm_mol.dihedrals[dih_idx].type.phi_k = 0.0
            logger.info(f'collecting QM and MM Energies for fragment {vi}')
            dihedrals = []
            mm_energies = []
            qm_energies = []
            conf_names = []
            for conf in self.all_conformer_files[vi]:
                conf_mol = rd_load_file(str(conf))
                positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
                mm_energy_dict = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=self.aqueous)
                parm_mol.positions = positions
                log_file = conf.with_suffix(".psi4.log")
                qm_energy = read_energy_from_log(log_file)
                dihedrals.append(float(parm_mol.dihedrals[dih_idx].measure() / 180.0 * np.pi))
                mm_energies.append(mm_energy_dict["total"])
                qm_energies.append(qm_energy * Hatree2kCalPerMol)
                conf_names.append(conf.stem)

            dihedrals = np.array(dihedrals)
            mm_energy = np.array(mm_energies)
            qm_energy = np.array(qm_energies)
            mm_energy = mm_energy - mm_energy.min()
            qm_energy = qm_energy - qm_energy.min()
            for idx, conf_name in enumerate(conf_names):
                self.conformer_data[vi][conf_name] = {
                        "dihedral": float(dihedrals[idx]),
                        "mm_energy": float(mm_energy[idx]),
                        "qm_energy": float(qm_energy[idx])
                    }
            optimizer = TorsionOptimizer(self.order, self.panelty_weight, self.threshold, fix_phase=self.fix_phase)
            optimizer.infer_parameters(
                dihedrals=dihedrals,
                energy_mm=mm_energy,
                energy_qm=qm_energy
            )
            self.parameter_set[vi] = optimizer.get_parameters()
            parent_dihedral_index = [vfrag.fragment_parent_mapping[a] for a in vfrag.pivotal_dihedral_quartet[0]]
            self.parameter_set[vi]["dihedral"] = parent_dihedral_index
            
        self.save_to_yaml(self.conformer_data, output_dir/"conformer_data.yaml")
        self.save_to_yaml(self.parameter_set, output_dir/"parameter_set.yaml")
            
    @staticmethod    
    def save_to_yaml(dict_data, yaml_file):
        with open(yaml_file, "w") as f:
            yaml.dump(dict_data, f)
        
    def write_script(self, slurm_filename, input_file, output_dir, charge, local=False):
        if local:
            slurm_content = "#!/bin/bash\n"
        else:
            slurm_content = SLURM_HEADER_CPU
        psi4_python_exec = Path(__file__).parent.parent / "qm" / "psi4_calculate_energy.py"
        slurm_content += f"python" \
                            f" {str(psi4_python_exec.resolve())}" \
                            f" {str(input_file)}" \
                            f" {str(output_dir)}" \
                            f" --charge {charge}" \
                            f" --memory '{self.memory}'" \
                            f" --n_threads {self.threads}" \
                            f" --method_basis {self.method}/{self.basis}" \
                            f"\n"
        with open(slurm_filename, "w") as f:
            f.write(slurm_content)
    
    def get_dihrdeal_terms_by_quartet(self, parm_mol, atom_index_list):
        idx_list = []
        atom_index_list = np.array(sorted(atom_index_list))
        for idx, dihedral in enumerate(parm_mol.dihedrals):
            _atom_index_list = np.array(sorted([dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]))
            if np.all(_atom_index_list == atom_index_list):
                idx_list.append(idx)
        return idx_list

    def write_script_aq(self, slurm_filename, input_file, output_dir, charge, local=False):
        if local:
            slurm_content = "#!/bin/bash\n"
        else:
            slurm_content = SLURM_HEADER_CPU
        orca_python_exec = Path(__file__).parent.parent / "qm" / "orca_calculate_energy.py"
        slurm_content += f"python" \
                            f" {str(orca_python_exec.resolve())}" \
                            f" {str(input_file)}" \
                            f" {str(output_dir)}" \
                            f" --charge {charge}" \
                            f" --n_threads {self.threads}" \
                            f" --method_basis {self.method}/{self.basis}" \
                            f" --aqueous" \
                            f"\n"
        with open(slurm_filename, "w") as f:
            f.write(slurm_content)
    
    

class AmberTorsionFactory(MoleculeFactory):
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
        self.order = order
        self.panelty_weight = panelty_weight
        self.threshold = threshold
        
        self.mol = None
        if template in ["A", "U", "G", "C"]:
            self.template_file = TEMPLATE_DIR / f"NA_{template}.pdb"
        elif template is not None:
            self.template_file = template
        else:
            self.template_file = None
        if self.template_file is not None:
            self.template_mol = rd_load_file(self.template_file)
        
        self.sugar_template_file = str(TEMPLATE_DIR/"sugar_template.pdb")
        self.sugar_template_mol = Chem.MolFromPDBFile(str(self.sugar_template_file), removeHs=False)
        
        self.scanning_steps = scanning_steps
        self.start = -180
        self.end = 180
        self.basis = basis
        self.method = method
    
    def load_file(self, filename, charge=None):
        """
        Load a molecule from a file.
        Input: 
            filename: str, path to the file. Supported formats are `PDB`, `XYZ`, `SDF`
                    if `PDB` or `XYZ` file is provided, charge must be provided.
            charge: int, charge of the molecule
        """
        file_path = Path(filename)
        if file_path.suffix in [".pdb", '.xyz'] and charge is None:
            logger.error("Please provide charge for pdb or xyz file")
            raise ValueError("Please provide charge for pdb or xyz file")
        if file_path.suffix in [".pdb", '.xyz']:
            determine_bond_order = True
        else:
            determine_bond_order = False
        self.mol = rd_load_file(file_path, removeHs=False, sanitize=True, charge=charge, determine_bond_order=determine_bond_order)
        
    def match_template(self):
        self.core_atoms = map_atoms(self.template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        self.sugar_fragment_mapping = map_atoms(self.sugar_template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        sugar_name_to_index = atomName_to_index(self.sugar_template_mol)
        mapping_dict = {}
        for atom in self.sugar_fragment_mapping:
            mapping_dict[atom[0]] = atom[1]
        self.sugar_fragment_mapping = mapping_dict
        self.O5p_index_parent = self.sugar_fragment_mapping[sugar_name_to_index["O5'"]]
        self.C5p_index_parent = self.sugar_fragment_mapping[sugar_name_to_index["C5'"]]
        self.C1p_index_parent = self.sugar_fragment_mapping[sugar_name_to_index["C1'"]]

    def fragmentize(self, output_dir):
        self.match_template()
        # n_atom_mod = self.mol.GetNumAtoms()
        # unique_atoms = [i for i in range(n_atom_mod) if i not in [a[1] for a in self.core_atoms]]

        TF = C5Fragmentizer(
                self.mol, 
                C1p_index=self.C1p_index_parent,
                O5p_index=self.O5p_index_parent,
                C5p_index=self.C5p_index_parent,)
        TF.fragmentize()
        self.fragment = TF.fragment
        dq_list = self.fragment.pivotal_dihedral_quartet
        if dq_list is None:
            raise RuntimeError("No pivotal dihedral quartet found.")
        dq = dq_list[0]
        self.fragment.save(str(output_dir), format="pdb")
        logger.info(f"Fragment  is saved to {str(output_dir)}")
                
    def gen(self, submit=False, local=False, overwrite=False):
        output_dir = Path(self.output_dir)
        logger.info(f" >>> Output directory: {output_dir} <<<")
        self.fragmentize(output_dir)
        self.slurm_files = []
        if self.fragment.pivotal_dihedral_quartet is None:
            raise RuntimeError("No pivotal dihedral quartet found.")
        logger.info(f"Scanning dihedral {self.fragment.pivotal_dihedral_quartet[0]}")
        dsc = DihedralScanner(
            input_file=str(output_dir/f"fragment.pdb"), 
            dihedrals=[self.fragment.pivotal_dihedral_quartet[0]],
            charge=0,
            workdir=str(output_dir),
            conformer_prefix=f"frag_conformer",
        ) 
        dsc.run(
            start=self.start,
            end=self.end + (360/self.scanning_steps)*(self.scanning_steps-1),
            steps=self.scanning_steps,
            overwrite=overwrite
        )
        self.conformer_files = dsc.get_conformers(self.fragment.pivotal_dihedral_quartet[0])
    
        for idx, conf in enumerate(self.conformer_files):
            slurm_file = (output_dir/f"conf{idx}.sh").resolve()
            if self.fragment.charge != 0:
                logger.warning(f"Fragment has charge {self.fragment.charge}")
                logger.warning(f"Please use appropriate QM methods for charged fragments")
            self.write_script(
                slurm_filename=slurm_file,
                input_file=str(conf),
                output_dir=(output_dir).resolve(),
                charge=self.fragment.charge,
                local=local
            )
            self.slurm_files.append(slurm_file)
            if not overwrite and os.path.exists(output_dir/f"{conf.stem}_property.txt"):
                logger.info(f"File {conf.stem}_property.txt exists. Skipping the calculation.")
                continue
                
            if local:
                cwd = Path.cwd().resolve()
                os.chdir(output_dir)
                logger.info(f"Running {slurm_file}")
                os.system(f"bash {slurm_file}")
                os.chdir(cwd)
            elif submit:
                cwd = Path.cwd().resolve()
                os.chdir(output_dir)
                logger.info(f"Submiting {slurm_file}")
                os.system(f"LLsub {slurm_file}")
                os.chdir(cwd)
                
        
    def optimize(self, overwrite=False):
        """
        Optimize the torsion parameters for the molecule.
        Input:
            overwrite: bool, if True, overwrite the existing files.
        Output:
            Save the conformer data and parameter set to the output directory.
        
        Description:
            1. Calculate atomic charges for each fragment.
            2. Parameterize each fragment.
            3. Collect QM and MM energies for each conformer.
            4. Optimize the torsion parameters.
        """
        output_dir = Path(self.output_dir)
        self.conformer_data = {}
        
        logger.info(f"Optimizing fragment")
        logger.info(f"calculating atomic charges for fragment")
        frag_dir = Path(output_dir)
        # get atomic charges
        charged_mol2 = frag_dir/f"{self.conformer_files[0].stem}.mol2"
        if overwrite or (not os.path.exists(charged_mol2)):
            RESP(
                str(self.conformer_files[0]),
                self.fragment.charge,
                str(frag_dir),
                self.mol_name,
                memory=self.memory, 
                n_threads=self.threads, 
                method_basis=f"{self.resp_method}/{self.resp_basis}",
                n_conformers=self.resp_n_conformers
            )
        else:
            logger.info(f"File exised, using existing charge file {charged_mol2}.")
        charged_pmd_mol = pmd.load_file(str(charged_mol2))
        logger.info("Parameterizing fragment")
        parm7 = frag_dir/f"{self.mol_name}_frag.parm7"
        if overwrite or (not os.path.exists(parm7)):
            self.parameterize(str(output_dir/"fragment.pdb"), 
                              frag_dir, prefix=f"{self.mol_name}_frag", addons=["set default PBRadii mbondi"])
        else:
            logger.info(f"File exised, using existing parameter file {parm7}.")
        parm_mol = pmd.load_file(str(parm7))
        for idx, atom in enumerate(parm_mol.atoms):
            atom.charge = charged_pmd_mol.atoms[idx].charge
        idx_list = self.get_dihrdeal_terms_by_quartet(parm_mol, self.fragment.pivotal_dihedral_quartet[0])
        for dih_idx in idx_list:
            parm_mol.dihedrals[dih_idx].type.phi_k = 0.0
        logger.info(f'collecting QM and MM Energies for fragment')
        dihedrals = []
        mm_energies = []
        qm_energies = []
        conf_names = []
        for conf in self.conformer_files:
            conf_mol = rd_load_file(str(conf))
            positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
            mm_energy_dict = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=True)
            parm_mol.positions = positions
            log_file = conf.parent/f"{conf.stem}_property.txt"
            qm_energy = read_energy_from_txt(log_file)
            dihedrals.append(float(parm_mol.dihedrals[dih_idx].measure() / 180.0 * np.pi))
            mm_energies.append(mm_energy_dict["total"])
            qm_energies.append(qm_energy * Hatree2kCalPerMol)
            conf_names.append(conf.stem)

        dihedrals = np.array(dihedrals)
        mm_energy = np.array(mm_energies)
        qm_energy = np.array(qm_energies)
        mm_energy = mm_energy - mm_energy.min()
        qm_energy = qm_energy - qm_energy.min()
        for idx, conf_name in enumerate(conf_names):
            self.conformer_data[conf_name] = {
                    "dihedral": float(dihedrals[idx]),
                    "mm_energy": float(mm_energy[idx]),
                    "qm_energy": float(qm_energy[idx])
                }
        optimizer = TorsionOptimizer(self.order, self.panelty_weight, self.threshold)
        optimizer.infer_parameters(
            dihedrals=dihedrals,
            energy_mm=mm_energy,
            energy_qm=qm_energy
        )
        self.parameter_set = optimizer.get_parameters()
        parent_dihedral_index = [self.fragment.fragment_parent_mapping[a] for a in self.fragment.pivotal_dihedral_quartet[0]]
        self.parameter_set["dihedral"] = parent_dihedral_index
        
        self.save_to_yaml(self.conformer_data, output_dir/"conformer_data.yaml")
        self.save_to_yaml(self.parameter_set, output_dir/"parameter_set.yaml")
            
    @staticmethod    
    def save_to_yaml(dict_data, yaml_file):
        with open(yaml_file, "w") as f:
            yaml.dump(dict_data, f)
        
    def write_script(self, slurm_filename, input_file, output_dir, charge, local=False):
        if local:
            slurm_content = "#!/bin/bash\n"
        else:
            slurm_content = SLURM_HEADER_CPU
        orca_python_exec = Path(__file__).parent.parent / "qm" / "orca_calculate_energy.py"
        slurm_content += f"python" \
                            f" {str(orca_python_exec.resolve())}" \
                            f" {str(input_file)}" \
                            f" {str(output_dir)}" \
                            f" --charge {charge}" \
                            f" --n_threads {self.threads}" \
                            f" --method_basis {self.method}/{self.basis}" \
                            f" --aqueous" \
                            f"\n"
        with open(slurm_filename, "w") as f:
            f.write(slurm_content)
    
    def get_dihrdeal_terms_by_quartet(self, parm_mol, atom_index_list):
        idx_list = []
        atom_index_list = np.array(sorted(atom_index_list))
        for idx, dihedral in enumerate(parm_mol.dihedrals):
            _atom_index_list = np.array(sorted([dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]))
            if np.all(_atom_index_list == atom_index_list):
                idx_list.append(idx)
        return idx_list
    
  
    
       



