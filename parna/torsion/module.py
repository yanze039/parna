from .optim import TorsionOptimizer
from .fragment import TorsionFragmentizer, C5Fragmentizer
from .conf import DihedralScanner
from parna.utils import (rd_load_file, map_atoms, SLURM_HEADER_CPU, 
                         atomName_to_index, inverse_mapping, select_from_list, 
                         get_suger_picker_angles_from_pseudorotation,
                         save_to_yaml)
from parna.constant import Hatree2kCalPerMol, PSEUDOROTATION
from parna.qm.psi4_utils import read_energy_from_log
from parna.qm.orca_utils import read_energy_from_txt
from parna.resp import RESP
from parna.molops import MoleculeFactory, SugarPatcher, modify_torsion_parameters
from parna.constant import DIHDEDRAL_CONSTRAINTS, DIHDEDRAL_CONSTRAINTS_PHOSPHATE
import parmed as pmd
from parna.logger import getLogger
from pathlib import Path
import numpy as np
import rdkit.Chem as Chem
from parna.resp import RESP_fragment
import os
import yaml
import copy



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
            fix_phase: bool = True,
            determine_bond_orders: bool = False,
            pairwise: bool = False
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
        self.determine_bond_orders = determine_bond_orders
        self.pairwise = pairwise
        
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
        TF = TorsionFragmentizer(self.mol, cap_methylation=self.cap_methylation, determine_bond_orders=self.determine_bond_orders)
        TF.fragmentize()
        self.valid_fragments = {}
        """
        self.valid_fragments = {
            idx: {
                "Inchi": str,
                "mol": Chem.Mol,
                "parent_rotatable_bond": list,
            }
        }
        """
        
        for idx, frag in enumerate(TF.fragments):
            dq = frag.pivotal_dihedral_quartet[0]
            if (np.any([frag.fragment_parent_mapping[a] in unique_atoms for a in dq])) or \
                        self.C1_index_parent in [frag.fragment_parent_mapping[a] for a in frag.rotatable_bond]:
                
                existed = False
                # avoid duplicate fragments with the same parent rotatable bond
                fragment_Inchi = Chem.MolToInchi(frag.mol)
                for vfrag in self.valid_fragments.values():
                    if fragment_Inchi == vfrag["Inchi"]:
                        frag_vfrag_mapping_list = map_atoms(frag.mol, vfrag["fragment"].mol)
                        frag_vfrag_mapping_dict = {}
                        for atom in frag_vfrag_mapping_list:
                            frag_vfrag_mapping_dict[atom[0]] = atom[1]
                        vfrag_frag_mapping_dict = inverse_mapping(frag_vfrag_mapping_dict)
                        _rotatable_bond_index = np.array(sorted([frag_vfrag_mapping_dict[a] for a in frag.rotatable_bond]))
                        if np.all(_rotatable_bond_index == np.array(sorted(vfrag["fragment"].rotatable_bond))):
                            existed = True
                            vfrag["parent_rotatable_bond"].append(
                                (frag.fragment_parent_mapping[frag.rotatable_bond[0]],
                                frag.fragment_parent_mapping[frag.rotatable_bond[1]])
                            )
                            vfrag["parent_dihedral_index"].append(
                                [frag.fragment_parent_mapping[vfrag_frag_mapping_dict[a]] for a in vfrag["fragment"].pivotal_dihedral_quartet[0]]
                            )
                            break
                if existed:
                    continue
                
                frag.save(str(output_dir/f"fragment{idx}"), format="pdb")
                logger.info(f"Rotatable bond: {frag.rotatable_bond} is valid for fragment {idx}. Parent image: {[frag.fragment_parent_mapping[a] for a in frag.rotatable_bond]}")
                logger.info(f"Fragment {idx} is saved to {str(output_dir/f'fragment{idx}')}")
                self.valid_fragments[idx] = {
                    "Inchi": fragment_Inchi,
                    "fragment": frag,
                    "parent_rotatable_bond": [(frag.fragment_parent_mapping[frag.rotatable_bond[0]], frag.fragment_parent_mapping[frag.rotatable_bond[1]])],
                    "parent_dihedral_index": [[frag.fragment_parent_mapping[a] for a in frag.pivotal_dihedral_quartet[0]], ]
                }
                #  parent_dihedral_index = [vfrag.fragment_parent_mapping[a] for a in vfrag.pivotal_dihedral_quartet[0]]
                     
        logger.debug(f"Valid fragments: {self.valid_fragments}")
                
    def gen(self, submit=False, local=False, overwrite=False):
        output_dir = Path(self.output_dir)
        logger.info(f" >>> Output directory: {output_dir} <<<")
        self.fragmentize(output_dir)
        self.all_conformer_files = {}
        self.slurm_files = []
        for vi, vfrag in self.valid_fragments.items():
            logger.info(f"Scanning dihedral {vfrag['fragment'].pivotal_dihedral_quartet[0]}")
            dsc = DihedralScanner(
                input_file=output_dir/f"fragment{vi}/fragment.pdb", 
                dihedrals=[vfrag["fragment"].pivotal_dihedral_quartet[0]],
                charge=vfrag["fragment"].charge,
                workdir=output_dir/f"fragment{vi}",
                conformer_prefix=f"frag_conformer",
            ) 
            dsc.run(
                start=self.start,
                end=self.end + (360/self.scanning_steps)*(self.scanning_steps-1),
                steps=self.scanning_steps,
                overwrite=overwrite
            )
            self.all_conformer_files[vi] = dsc.conformers[vfrag["fragment"].pivotal_dihedral_quartet[0]]
        
            for idx, conf in enumerate(dsc.conformers[vfrag["fragment"].pivotal_dihedral_quartet[0]]):
                slurm_file = (output_dir/f"fragment{vi}/conf{idx}.slurm").resolve()
                if vfrag["fragment"].charge != 0:
                    logger.warning(f"Fragment {vi} has charge {vfrag['fragment'].charge}")
                    logger.warning(f"Please use appropriate QM methods for charged fragments")
                self.write_script(
                    slurm_filename=slurm_file,
                    input_file=str(conf),
                    output_dir=(output_dir/f"fragment{vi}").resolve(),
                    charge=vfrag["fragment"].charge,
                    local=local,
                    aqueous=self.aqueous
                )
                self.slurm_files.append(slurm_file)
                
                if not overwrite and ((os.path.exists(conf.with_suffix(".psi4.log")) and not self.aqueous) \
                    or (os.path.exists(conf.with_suffix(".molden")) and self.aqueous)):
                    continue
                
                cwd = Path.cwd().resolve()
                os.chdir(output_dir)
                if local:
                    logger.info(f"Running {slurm_file}")
                    os.system(f"bash {slurm_file}")
                elif submit:
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
                    vfrag["fragment"].charge, frag_dir
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
            idx_list = self.get_dihrdeal_terms_by_quartet(parm_mol, vfrag["fragment"].pivotal_dihedral_quartet[0])
            assert len(idx_list) > 0, f"No dihedral term found for fragment {vi}." 
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
                mm_energy = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=self.aqueous)
                parm_mol.positions = positions
                if self.aqueous:
                    log_file = conf.parent/f"{conf.stem}_property.txt"
                    qm_energy = read_energy_from_txt(log_file)
                else:
                    log_file = conf.with_suffix(".psi4.log")
                    qm_energy = read_energy_from_log(log_file)
                dihedrals.append(float(parm_mol.dihedrals[idx_list[0]].measure() / 180.0 * np.pi))
                mm_energies.append(mm_energy)
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
                energy_qm=qm_energy,
                pairwise=self.pairwise
            )
            self.parameter_set[vi] = optimizer.get_parameters()
            # parent_dihedral_index = [vfrag.fragment_parent_mapping[a] for a in vfrag.pivotal_dihedral_quartet[0]]
            # self.parameter_set[vi]["dihedral"] = parent_dihedral_index
            self.parameter_set[vi]["dihedral"] = vfrag["parent_dihedral_index"]
            
            
        save_to_yaml(self.conformer_data, output_dir/"conformer_data.yaml")
        save_to_yaml(self.parameter_set, output_dir/"parameter_set.yaml")
    
    def write_script(self, slurm_filename, input_file, output_dir, charge, local=False, aqueous=False):
        if aqueous:
            self.write_script_aq(slurm_filename, input_file, output_dir, charge, local)
        else:
            self.write_script_gas(slurm_filename, input_file, output_dir, charge, local)
        
    def write_script_gas(self, slurm_filename, input_file, output_dir, charge, local=False):
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

        TF = C5Fragmentizer(
                self.mol, 
                C1p_index=self.C1p_index_parent,
                O5p_index=self.O5p_index_parent,
                C5p_index=self.C5p_index_parent,
        )
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
            charge=self.fragment.charge,
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
            mm_energy = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=True)
            parm_mol.positions = positions
            log_file = conf.parent/f"{conf.stem}_property.txt"
            qm_energy = read_energy_from_txt(log_file)
            dihedrals.append(float(parm_mol.dihedrals[dih_idx].measure() / 180.0 * np.pi))
            mm_energies.append(mm_energy)
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
        
        save_to_yaml(self.conformer_data, output_dir/"conformer_data.yaml")
        save_to_yaml(self.parameter_set, output_dir/"parameter_set.yaml")
        
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
    
    
class NucleotideTorsionFactory(MoleculeFactory):
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
            fix_phase: bool = True,
            determine_bond_orders: bool = False,
            pairwise: bool = False
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
        self.determine_bond_orders = determine_bond_orders
        self.pairwise = pairwise
        
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
        
    def match_template(self):
        r"""Match the template molecule with the input molecule.
        Two template molecules are used:
            1. The nucleotide template molecule.
            2. The sugar template molecule.
            
        """
        self.core_atoms = map_atoms(self.template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        self.sugar_fragment_mapping = map_atoms(self.sugar_template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        sugar_name_to_index = atomName_to_index(self.sugar_template_mol)
        mapping_dict = {}
        for atom in self.sugar_fragment_mapping:
            mapping_dict[atom[0]] = atom[1]
        self.sugar_fragment_mapping = mapping_dict
        _atom_names = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'","O4'", "O5'", "HO3'", "HO2'", "HO5'"]
        self.parent_suger_symbol_index = {}
        for atom_name in _atom_names:
            if sugar_name_to_index[atom_name] in self.sugar_fragment_mapping:
                self.parent_suger_symbol_index[atom_name] = self.sugar_fragment_mapping[sugar_name_to_index[atom_name]]
            
    def fragmentize(self, output_dir):
        
        self.match_template()
        n_atom_mod = self.mol.GetNumAtoms()
        unique_atoms = [i for i in range(n_atom_mod) if i not in [a[1] for a in self.core_atoms]]
        TF = TorsionFragmentizer(self.mol, cap_methylation=self.cap_methylation, determine_bond_orders=self.determine_bond_orders)
        TF.fragmentize()
        self.valid_fragments = {}
        """
        self.valid_fragments = {
            idx: {
                "Inchi": str,
                "mol": Chem.Mol,
                "parent_rotatable_bond": list,
            }
        }
        """
        
        for idx, frag in enumerate(TF.fragments):
            dq = frag.pivotal_dihedral_quartet[0]
            if (np.any([frag.fragment_parent_mapping[a] in unique_atoms for a in dq])) or \
                        self.parent_suger_symbol_index["C1'"] in [frag.fragment_parent_mapping[a] for a in frag.rotatable_bond]:
                
                existed = False
                # avoid duplicate fragments with the same parent rotatable bond
                fragment_Inchi = Chem.MolToInchi(frag.mol)
                for vfrag in self.valid_fragments.values():
                    if fragment_Inchi == vfrag["Inchi"]:
                        frag_vfrag_mapping_list = map_atoms(frag.mol, vfrag["fragment"].mol)
                        frag_vfrag_mapping_dict = {}
                        for atom in frag_vfrag_mapping_list:
                            frag_vfrag_mapping_dict[atom[0]] = atom[1]
                        vfrag_frag_mapping_dict = inverse_mapping(frag_vfrag_mapping_dict)
                        _rotatable_bond_index = np.array(sorted([frag_vfrag_mapping_dict[a] for a in frag.rotatable_bond]))
                        if np.all(_rotatable_bond_index == np.array(sorted(vfrag["fragment"].rotatable_bond))):
                            existed = True
                            vfrag["parent_rotatable_bond"].append(
                                (frag.fragment_parent_mapping[frag.rotatable_bond[0]],
                                frag.fragment_parent_mapping[frag.rotatable_bond[1]])
                            )
                            vfrag["parent_dihedral_index"].append(
                                [frag.fragment_parent_mapping[vfrag_frag_mapping_dict[a]] for a in vfrag["fragment"].pivotal_dihedral_quartet[0]]
                            )
                            break
                if existed:
                    continue
                
                frag.save(str(output_dir/f"fragment{idx}"), format="pdb")
                logger.info(f"Rotatable bond: {frag.rotatable_bond} is valid for fragment {idx}. Parent image: {[frag.fragment_parent_mapping[a] for a in frag.rotatable_bond]}")
                logger.info(f"Fragment {idx} is saved to {str(output_dir/f'fragment{idx}')}")
                self.valid_fragments[idx] = {
                    "Inchi": fragment_Inchi,
                    "fragment": frag,
                    "parent_rotatable_bond": [(frag.fragment_parent_mapping[frag.rotatable_bond[0]], frag.fragment_parent_mapping[frag.rotatable_bond[1]])],
                    "parent_dihedral_index": [[frag.fragment_parent_mapping[a] for a in frag.pivotal_dihedral_quartet[0]], ],
                    "fragment_constraints": {}
                }
                # DIHDEDRAL_CONSTRAINTS
                for cname, constraints in DIHDEDRAL_CONSTRAINTS.items():
                    atom_symbols = constraints["atoms"]
                    try:
                        if np.all([self.parent_suger_symbol_index[a] in frag.parent_fragment_mapping for a in atom_symbols]):
                            self.valid_fragments[idx]["fragment_constraints"][cname] = {
                                "type": "dihedral",
                                "atom_index": [frag.parent_fragment_mapping[self.parent_suger_symbol_index[a]] for a in atom_symbols],
                                "value": constraints["angle"],
                            }
                    except KeyError as e:
                        print(e)
                        continue
                
        logger.debug(f"Valid fragments: {self.valid_fragments}")
                
    def gen(self, submit=False, local=False, overwrite=False):
        output_dir = Path(self.output_dir)
        logger.info(f" >>> Output directory: {output_dir} <<<")
        self.fragmentize(output_dir)
        self.all_conformer_files = {}
        self.slurm_files = []
        for vi, vfrag in self.valid_fragments.items():
            logger.info(f"Scanning dihedral {vfrag['fragment'].pivotal_dihedral_quartet[0]}")
            dsc = DihedralScanner(
                input_file=output_dir/f"fragment{vi}/fragment.pdb", 
                dihedrals=[vfrag["fragment"].pivotal_dihedral_quartet[0]],
                charge=vfrag["fragment"].charge,
                workdir=output_dir/f"fragment{vi}",
                conformer_prefix=f"frag_conformer",
                constraints=[[x["atom_index"], x["value"]] for x in vfrag["fragment_constraints"].values()],
                force_constant=1.0  # Hartree/(Bohr**2)
            ) 

            dsc.run(
                start=self.start,
                # end=self.end + (360/self.scanning_steps)*(self.scanning_steps-1),
                end=self.start + (360/self.scanning_steps)*(self.scanning_steps-1),
                steps=self.scanning_steps,
                overwrite=overwrite
            )
            self.all_conformer_files[vi] = dsc.conformers[vfrag["fragment"].pivotal_dihedral_quartet[0]]
        
            for idx, conf in enumerate(dsc.conformers[vfrag["fragment"].pivotal_dihedral_quartet[0]]):
                slurm_file = (output_dir/f"fragment{vi}/conf{idx}.slurm").resolve()
                if vfrag["fragment"].charge != 0:
                    logger.warning(f"Fragment {vi} has charge {vfrag['fragment'].charge}")
                    logger.warning(f"Please use appropriate QM methods for charged fragments")
                self.write_script(
                    slurm_filename=slurm_file,
                    input_file=str(conf),
                    output_dir=(output_dir/f"fragment{vi}").resolve(),
                    charge=vfrag["fragment"].charge,
                    local=local,
                    aqueous=self.aqueous
                )
                self.slurm_files.append(slurm_file)
                
                if not overwrite and ((os.path.exists(conf.with_suffix(".psi4.log")) and not self.aqueous) \
                    or (os.path.exists(conf.with_suffix(".molden")) and self.aqueous)):
                    continue
                
                cwd = Path.cwd().resolve()
                os.chdir(output_dir)
                if local:
                    logger.info(f"Running {slurm_file}")
                    os.system(f"bash {slurm_file}")
                elif submit:
                    logger.info(f"Submiting {slurm_file}")
                    os.system(f"LLsub {slurm_file}")
                os.chdir(cwd)
        
    def optimize(self, overwrite=False):
        """Optimize the torsion parameters for the molecule.
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
            charged_mol2 = frag_dir/f"{self.all_conformer_files[vi][0].stem}.resp.mol2"
            if overwrite or (not os.path.exists(charged_mol2)):
                self.charge_molecule(
                    select_from_list(self.all_conformer_files[vi], self.resp_n_conformers, method="even"),
                    vfrag["fragment"].charge, frag_dir
                )
            else:
                logger.info(f"File exised, using existing charge file {charged_mol2}.")
            charged_pmd_mol = pmd.load_file(str(charged_mol2))
            logger.info("Parameterizing fragment")
            parm7 = frag_dir/f"{self.mol_name}_frag{vi}.parm7"
            if self.aqueous:
                addons = ["set default PBRadii mbondi"]
            else:
                addons = []
            self.parameterize(str(output_dir/f"fragment{vi}/fragment.pdb"), frag_dir, prefix=f"{self.mol_name}_frag{vi}", addons=addons)
          
            parm_mol = pmd.load_file(str(parm7))
            for idx, atom in enumerate(parm_mol.atoms):
                atom.charge = charged_pmd_mol.atoms[idx].charge
            
            # START  >>> turn off the dihedral terms along the rotatable bond <<<
            # idx_list_dihedral = self.get_dihrdeal_terms_by_quartet(parm_mol, vfrag["fragment"].pivotal_dihedral_quartet[0])
            idx_list_bond = self.get_dihedral_terms_by_bond(parm_mol, [vfrag["fragment"].pivotal_dihedral_quartet[0][1], vfrag["fragment"].pivotal_dihedral_quartet[0][2]])
                        
            logger.info(f"Found {len(idx_list_bond)} dihedral terms by bonds: {idx_list_bond}")
            # assert len(idx_list_dihedral) > 0, f"No dihedral term found by dihedral quartet for fragment {idx_list_dihedral}." 
            # assert np.all([idx in idx_list_bond for idx in idx_list_dihedral]), f"Index mismatch between dihedral and bond terms: {idx_list_dihedral} and {idx_list_bond}"
            for dih_idx in idx_list_bond:
                parm_mol.dihedrals[dih_idx].type.phi_k = 0.0
            # END    >>> turn off the dihedral terms along the rotatable bond <<<
            
            logger.info(f'collecting QM and MM Energies for fragment {vi}')
            dihedrals = {}
            mm_energies = []
            qm_energies = []
            conf_names = []

            for conf in self.all_conformer_files[vi]:
                # >> read QM energy
                if self.aqueous:
                    log_file = conf.parent/f"{conf.stem}_property.txt"
                    qm_energy = read_energy_from_txt(log_file)
                else:
                    log_file = conf.with_suffix(".psi4.log")
                    qm_energy = read_energy_from_log(log_file)
                # >> calculate MM energy
                conf_mol = rd_load_file(str(conf))
                # unit from RDkit is Angstrom, convert to nm by dividing 10.
                positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
                parm_mol.positions = positions
                # >> NEW VERSION [ALL dihedral terms were considered]
                
                mm_restraints = []
                dihedral_each_conf = {}

                for dihedral_atom_index in vfrag["fragment"].all_dihedral_quartets:
                    dihedral_degrees = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, dihedral_atom_index)[0]].measure())
                    dihedral_each_conf[dihedral_atom_index] = dihedral_degrees / 180.0 * np.pi
                    mm_restraints.append({"atom_index": dihedral_atom_index, "value": dihedral_degrees})
                mm_restraints.extend([x for x in vfrag["fragment_constraints"].values()])
                # >> NEW VERSION END
                
                # >> OLD VERSION [ONLY one dihedral term was considered]
                # dihedral_degrees = float(parm_mol.dihedrals[idx_list_dihedral[0]].measure())
                # dihedrals.append(dihedral_degrees / 180.0 * np.pi)
                # mm_restraints = [{"atom_index": vfrag["fragment"].pivotal_dihedral_quartet[0], "value": dihedral_degrees}]
                # mm_restraints.extend([x for x in vfrag["fragment_constraints"].values()])
                # >> OLD VERSION END
                
                mm_energy = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=self.aqueous,
                                                     optimize=True, 
                                                     restraints=mm_restraints,
                                                     save=False, output_file=conf.parent/f"{conf.stem}_mm_optimized.pdb")
                dihedrals[conf.stem] = dihedral_each_conf
                mm_energies.append(mm_energy)
                qm_energies.append(qm_energy * Hatree2kCalPerMol)
                conf_names.append(conf.stem)

            # dihedrals = np.array(dihedrals)
            mm_energy = np.array(mm_energies)
            qm_energy = np.array(qm_energies)
            mm_energy = mm_energy - mm_energy.min()
            qm_energy = qm_energy - qm_energy.min()
            for idx, conf_name in enumerate(conf_names):
                self.conformer_data[vi][conf_name] = {
                        # "dihedral": float(dihedrals[idx]),
                        "dihedral": dihedrals[conf_name],
                        "mm_energy": float(mm_energy[idx]),
                        "qm_energy": float(qm_energy[idx])
                    }
            optimizer = TorsionOptimizer(self.order, self.panelty_weight, self.threshold, fix_phase=self.fix_phase, n_dihedrals=len(vfrag["fragment"].all_dihedral_quartets))
            # of shape [n_conformers, n_dihedrals]
            dihedral_value_matrix = np.array([[dihedrals[conf_name][x] for x in vfrag["fragment"].all_dihedral_quartets] for conf_name in conf_names])
            
            optimizer.infer_parameters(
                dihedrals=dihedral_value_matrix,
                energy_mm=mm_energy,
                energy_qm=qm_energy,
                pairwise=self.pairwise
            )
            self.parameter_set[vi] = optimizer.get_parameters()
            # self.parameter_set[vi]["dihedral"] = vfrag["parent_dihedral_index"]
            self.parameter_set[vi]["dihedral"] = [[vfrag["fragment"].fragment_parent_mapping[xi] for xi in x] for x in vfrag["fragment"].all_dihedral_quartets]

        save_to_yaml(self.conformer_data, output_dir/"conformer_fragment.yaml")
        save_to_yaml(self.parameter_set, output_dir/"parameter_fragment.yaml")
    


class EpsilonZetaTorsionFactory(MoleculeFactory):
    def __init__(self, 
                order: int = 4, 
                panelty_weight: float = 0.1, 
                threshold: float = 0.01, 
                mol_name: str = "mol",
                atom_type: str = "amber",
                method: str = "wB97X-V",
                basis: str = "def2-TZVPD",
                threads: int = 48,
                memory: str = "160 GB",
                resp_method: str = "HF",
                resp_basis: str = "6-31+G*",
                resp_n_conformers: int = 6,
                output_dir: str = None,
                cap_methylation: bool = True,
                aqueous: bool = False,
                fix_phase: bool = True,
                determine_bond_orders: bool = False,
                pairwise: bool = False,
                constrain_sugar_pucker: bool = True,
                epsilon_grids = [0, 35, 80, 125] + list(range(170, 290, 15)) + [290, 335],  # 14
                zeta_grids = [0, 40, 85, 130, 175] + list(range(220, 340, 15)) + [340],  # 14
                phosphate_style: str = "amber",
                alpha_gamma_style: str = "bsc0"
        ):
       
        self.basis = basis
        self.method = method
        self.order = order
        self.panelty_weight = panelty_weight
        self.threshold = threshold
        self.threads = threads
        self.memory = memory
        self.resp_method = resp_method
        self.resp_basis = resp_basis
        self.resp_n_conformers = resp_n_conformers
        self.output_dir = output_dir
        self.cap_methylation = cap_methylation
        self.aqueous = aqueous
        self.fix_phase = fix_phase
        self.determine_bond_orders = determine_bond_orders
        self.pairwise = pairwise
        self.mol_name = mol_name
        self.atom_type = atom_type
        self.constrain_sugar_pucker = constrain_sugar_pucker
        self.epsilon_grids = epsilon_grids
        self.zeta_grids = zeta_grids
        self.phosphate_style = phosphate_style
        self.alpha_gamma_style = alpha_gamma_style
        
    
    def match_template(self):
        r"""Match the template molecule with the input molecule.
        Two template molecules are used:
            1. The nucleotide template molecule.
            2. The sugar template molecule.
            
        """
        
        self.template_file = TEMPLATE_DIR / "NA_A.pdb"
        self.template_mol = rd_load_file(self.template_file)
        self.sugar_template_file = str(TEMPLATE_DIR/"sugar_template.pdb")
        self.sugar_template_mol = Chem.MolFromPDBFile(str(self.sugar_template_file), removeHs=False)
        self.sugar_template_with_PO3_file = str(TEMPLATE_DIR/"sugar_template_with_3p_PO3.pdb")
        self.sugar_template_with_PO3_mol = Chem.MolFromPDBFile(str(self.sugar_template_with_PO3_file), removeHs=False)
        self.mpatcher = SugarPatcher(self.mol)
        self.patched_mol = self.mpatcher.patch_three_prime_end()
        
        self.sugar_fragment_mapping = map_atoms(
            self.sugar_template_with_PO3_mol, 
            self.patched_mol, 
            ringMatchesRingOnly=False, 
            completeRingsOnly=False,
        )
        
        sugar_tmpl_name_to_index = atomName_to_index(self.sugar_template_with_PO3_mol)
        mapping_dict = {}
        for atom in self.sugar_fragment_mapping:
            mapping_dict[atom[0]] = atom[1]
        self.sugar_fragment_mapping = mapping_dict
        _atom_names = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'","O4'", "O5'", "HO2'", "HO5'", "P", "OP1", "OP2", "O52", "C01"]
        self.mol_suger_symbol_index = {}
        for atom_name in _atom_names:
            if sugar_tmpl_name_to_index[atom_name] in self.sugar_fragment_mapping:
                self.mol_suger_symbol_index[atom_name] = self.sugar_fragment_mapping[sugar_tmpl_name_to_index[atom_name]]
        
        # tag necessary atoms
        tagged_atoms = []
        
        for atom in self.patched_mol.GetAtoms():
            if atom.GetProp("_PatchAtom") == "1":
                tagged_atoms.append(atom.GetIdx())
        
        if "O2'" in self.mol_suger_symbol_index:
            tagged_atoms.append(self.mol_suger_symbol_index["O2'"])
            for neighbor in self.patched_mol.GetAtomWithIdx(self.mol_suger_symbol_index["O2'"]).GetNeighbors():
                tagged_atoms.append(neighbor.GetIdx())
        else:
            aggressive_mapping = map_atoms(
                self.sugar_template_with_PO3_mol, 
                self.patched_mol, 
                ringMatchesRingOnly=False, 
                completeRingsOnly=False,
                atomCompare=Chem.rdFMCS.AtomCompare.CompareAny
            )
            if sugar_tmpl_name_to_index[atom_name] in aggressive_mapping:
                tagged_atoms.append(aggressive_mapping[sugar_tmpl_name_to_index[atom_name]])
                for neighbor in self.patched_mol.GetAtomWithIdx(aggressive_mapping[sugar_tmpl_name_to_index[atom_name]]).GetNeighbors():
                    tagged_atoms.append(neighbor.GetIdx())
            else:
                raise RuntimeError("O2' atom not found.")
        
        for ta in tagged_atoms:
            self.patched_mol.GetAtomWithIdx(ta).SetProp("_keep_at_fragment", "1")
            
    def fragmentize(self, output_dir):
        """Fragmentize the molecule by O3'-P bond.
        """
        self.match_template()
        EZTF = TorsionFragmentizer(
                self.patched_mol, 
                cap_methylation=False, 
                determine_bond_orders=False,
                break_aromatic_ring=True
            )
        frag = EZTF.fragment_on_bond(
            self.mol_suger_symbol_index["O3'"],
            self.mol_suger_symbol_index["P"],
            
        )
        if frag is None:
            logger.error(f"Fragmentization failed for molecule {self.mol_name}")
        frag.save(str(output_dir/f"fragment_epsilon_zeta"), format="pdb")
        logger.info(f"Rotatable bond: Epsilon/Zeta is valid for fragment ")
        logger.info(f"Fragment is saved to {str(output_dir/f'fragment_epsilon_zeta')}")
        self.valid_fragment = {
            "fragment": frag,
            "epsilon": tuple(frag.parent_fragment_mapping[self.mol_suger_symbol_index[x]] for x in ["C4'", "C3'", "O3'", "P"]),
            "zeta": tuple(frag.parent_fragment_mapping[self.mol_suger_symbol_index[x]] for x in ["C3'", "O3'", "P", "O52"]),
            "fragment_constraints": {}
        }
        # DIHDEDRAL_CONSTRAINTS
        if self.constrain_sugar_pucker:
            for ctype in ["C3'-endo-constraint-v1", "C3'-endo-constraint-v3"]:
                constraints = DIHDEDRAL_CONSTRAINTS[ctype]
                atom_symbols = constraints["atoms"]
                self.valid_fragment["fragment_constraints"][ctype] = {
                    "type": "dihedral",
                    "atom_index": [frag.parent_fragment_mapping[self.mol_suger_symbol_index[a]] for a in atom_symbols],
                    "value": constraints["angle"],
                }
            for ctype in ["O2'-constraint", "alpha"]:
                constraints = DIHDEDRAL_CONSTRAINTS_PHOSPHATE[ctype]
                atom_symbols = constraints["atoms"]
                self.valid_fragment["fragment_constraints"][ctype] = {
                    "type": "dihedral",
                    "atom_index": [frag.parent_fragment_mapping[self.mol_suger_symbol_index[a]] for a in atom_symbols],
                    "value": constraints["angle"],
                }
    
    def gen(self, submit=False, local=False, overwrite=False):
        output_dir = Path(self.output_dir)
        logger.info(f" >>> Output directory: {output_dir} <<<")
        self.fragmentize(output_dir)
        self.all_conformer_files = []
        self.slurm_files = []
        
        logger.info(f"Scanning dihedral Epsilon / Zeta")
        
        # >>> 2D scanning: Epsilon / Zeta <<<
        for ep in self.epsilon_grids:
            dsc = DihedralScanner(
                input_file=str(output_dir/f"fragment_epsilon_zeta/fragment.pdb"), 
                dihedrals=[self.valid_fragment["zeta"]],
                charge=self.valid_fragment["fragment"].charge,
                workdir=str(output_dir/f"fragment_epsilon_zeta"),
                conformer_prefix=f"frag_conformer_ep_{int(ep):03d}",
                constraints=[[x["atom_index"], x["value"]] for x in self.valid_fragment["fragment_constraints"].values()]\
                    + [[self.valid_fragment["epsilon"], ep]],
                force_constant=1.0,  # Hartree/(Bohr**2)
                warming_constraints=True
            ) 

            dsc.run_on_grids(
                self.zeta_grids,
                overwrite=overwrite
            )
            self.all_conformer_files.extend(dsc.conformers[self.valid_fragment["zeta"]])
    
        for idx, conf in enumerate(self.all_conformer_files):
            slurm_file = (output_dir/f"fragment_epsilon_zeta/conf{idx}.slurm").resolve()
            if self.valid_fragment["fragment"].charge != 0:
                logger.warning(f"Fragment has charge {self.valid_fragment['fragment'].charge}")
                logger.warning(f"Please use appropriate QM methods for charged fragments")
            self.write_script(
                slurm_filename=slurm_file,
                input_file=str(conf),
                output_dir=(output_dir/f"fragment_epsilon_zeta").resolve(),
                charge=self.valid_fragment["fragment"].charge,
                local=local,
                aqueous=self.aqueous
            )
            self.slurm_files.append(slurm_file)
            
            if not overwrite and ((os.path.exists(conf.with_suffix(".psi4.log")) and not self.aqueous) \
                or (os.path.exists(conf.with_suffix(".molden")) and self.aqueous)):
                continue
            
            cwd = Path.cwd().resolve()
            os.chdir(output_dir)
            if local:
                logger.info(f"Running {slurm_file}")
                os.system(f"bash {slurm_file}")
                os.remove(slurm_file)
            elif submit:
                logger.info(f"Submiting {slurm_file}")
                os.system(f"LLsub {slurm_file}")
            for suffix in [".gbw", ".cpcm", ".densities"]:
                if os.path.exists(conf.with_suffix(suffix)):
                    os.remove(conf.with_suffix(suffix))
            os.chdir(cwd)
            
            
    def optimize(self, overwrite=False, suffix=""):
        output_dir = Path(self.output_dir)
        logger.info(f"Optimizing fragment Epsilon / Zeta")
        logger.info(f"calculating atomic charges for fragment Epsilon / Zeta")
        frag_dir = output_dir/f"fragment_epsilon_zeta"
        # get atomic charges
        # >>>    CHARGE    <<<
        charged_mol2 = frag_dir/f"{self.all_conformer_files[0].stem}.resp.mol2"
        if overwrite or (not os.path.exists(charged_mol2)):
            RESP_fragment(
                select_from_list(self.all_conformer_files, self.resp_n_conformers, method="even"),
                self.valid_fragment["fragment"].charge,
                str(frag_dir),
                self.mol_name,
                memory=self.memory, 
                n_threads=self.threads, 
                method_basis=f"{self.resp_method}/{self.resp_basis}",
            )
        else:
            logger.info(f"File exised, using existing charge file {charged_mol2}.")
        charged_pmd_mol = pmd.load_file(str(charged_mol2))
        logger.info("Parameterizing fragment")
        parm7 = frag_dir/f"{self.mol_name}_frag_epsilon_zeta.parm7"
        
        if self.aqueous:
            addons = ["set default PBRadii mbondi"]
        else:
            addons = []
        self.parameterize(str(output_dir/"fragment_epsilon_zeta/fragment.pdb"), 
                          frag_dir, prefix=f"{self.mol_name}_frag_epsilon_zeta", 
                          addons=addons,  mod_atom_types={
                              self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["OP1"]]: "OP",
                              self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["OP2"]]: "OP",
                              self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["O3'"]]: "OR",
                              self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["O52"]]: "OR",
                              self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["C01"]]: "CI",
                          })
        
        parm_mol = pmd.load_file(str(parm7))
        for idx, atom in enumerate(parm_mol.atoms):
            atom.charge = charged_pmd_mol.atoms[idx].charge
        
        # START  >>> turn off the dihedral terms along the rotatable bond <<<
        idx_list_epsilon = self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["epsilon"])
        idx_list_zeta = self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["zeta"])
        assert len(idx_list_epsilon) > 0, f"No dihedral term found for fragment Epsilon." 
        assert len(idx_list_zeta) > 0, f"No dihedral term found for fragment Zeta."
        for idx_list in [idx_list_epsilon, idx_list_zeta]:
            for dih_idx in idx_list:
                parm_mol.dihedrals[dih_idx].type.phi_k = 0.0
        logger.info(f'collecting QM and MM Energies for fragment Epsilon / Zeta')
        
        dihedrals = {}
        mm_energies = []
        qm_energies = []
        conf_names = []

        for conf in self.all_conformer_files:
            # >> read QM energy
            if self.aqueous:
                log_file = conf.parent/f"{conf.stem}_property.txt"
                qm_energy = read_energy_from_txt(log_file)
            else:
                log_file = conf.with_suffix(".psi4.log")
                qm_energy = read_energy_from_log(log_file)
            # >> calculate MM energy
            conf_mol = rd_load_file(str(conf))
            # unit from RDkit is Angstrom, convert to nm by dividing 10.
            positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
            
            parm_mol.positions = positions
            # >> NEW VERSION [ALL dihedral terms were considered]
            
            mm_restraints = []
            dihedral_each_conf = {}

            dihedral_degree_epsilon = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["epsilon"])[0]].measure())
            dihedral_degree_zeta = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["zeta"])[0]].measure())
            dihedral_each_conf["epsilon"] = dihedral_degree_epsilon / 180.0 * np.pi
            dihedral_each_conf["zeta"] = dihedral_degree_zeta / 180.0 * np.pi
            mm_restraints.append({"atom_index": self.valid_fragment["epsilon"], "value": dihedral_degree_epsilon})
            mm_restraints.append({"atom_index": self.valid_fragment["zeta"], "value": dihedral_degree_zeta})
            mm_restraints.extend([x for x in self.valid_fragment["fragment_constraints"].values()])
            logger.info(conf.stem)
            
            mm_energy = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=self.aqueous,
                                                    optimize=True, 
                                                    restraints=mm_restraints,
                                                    save=False, 
                                                    output_file=f"{conf.stem}_mm_opmized.pdb")
                                                    # output_file=conf.parent/f"{conf.stem}_mm_optimized.pdb")
            dihedrals[conf.stem] = dihedral_each_conf
            mm_energies.append(mm_energy)
            qm_energies.append(qm_energy * Hatree2kCalPerMol)
            conf_names.append(conf.stem)

        mm_energy = np.array(mm_energies)
        qm_energy = np.array(qm_energies)
        mm_energy = mm_energy - mm_energy.min()
        qm_energy = qm_energy - qm_energy.min()
        self.conformer_data = {}
        for idx, conf_name in enumerate(conf_names):
            self.conformer_data[conf_name] = {
                    # "dihedral": float(dihedrals[idx]),
                    "dihedral": dihedrals[conf_name],
                    "mm_energy": float(mm_energy[idx]),
                    "qm_energy": float(qm_energy[idx])
                }
        optimizer = TorsionOptimizer(self.order, self.panelty_weight, self.threshold, fix_phase=self.fix_phase, n_dihedrals=2)
        # of shape [n_conformers, n_dihedrals]
        dihedral_value_matrix = np.array([[dihedrals[conf_name][x] for x in ["epsilon", "zeta"]] for conf_name in conf_names])
        
        optimizer.infer_parameters(
            dihedrals=dihedral_value_matrix,
            energy_mm=mm_energy,
            energy_qm=qm_energy,
            pairwise=self.pairwise
        )
        self.parameter_set = optimizer.get_parameters()

        self.parameter_set["dihedral"] = ["epsilon", "zeta"]
        
        save_to_yaml(self.conformer_data, output_dir/f"conformer_epsilon_zeta{suffix}.yaml")
        save_to_yaml(self.parameter_set, output_dir/f"parameter_epsilon_data{suffix}.yaml")
    
    def getCMAP(self, overwrite=False, suffix=""):
        output_dir = Path(self.output_dir)
        logger.info(f"Optimizing fragment Epsilon / Zeta")
        logger.info(f"calculating atomic charges for fragment Epsilon / Zeta")
        frag_dir = output_dir/f"fragment_epsilon_zeta"
        # get atomic charges
        # >>>    CHARGE    <<<
        charged_mol2 = frag_dir/f"{self.all_conformer_files[0].stem}.resp.mol2"
        if overwrite or (not os.path.exists(charged_mol2)):
            RESP_fragment(
                select_from_list(self.all_conformer_files, self.resp_n_conformers, method="even"),
                self.valid_fragment["fragment"].charge,
                str(frag_dir),
                self.mol_name,
                memory=self.memory, 
                n_threads=self.threads, 
                method_basis=f"{self.resp_method}/{self.resp_basis}",
            )
        else:
            logger.info(f"File exised, using existing charge file {charged_mol2}.")
        charged_pmd_mol = pmd.load_file(str(charged_mol2))
        logger.info("Parameterizing fragment")
        
        if self.aqueous:
            addons = ["set default PBRadii mbondi"]
        else:
            addons = []
        
        modified_atom_types = {}
        if self.phosphate_style == "dcase":
            modified_atom_types.update(
                {
                    self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["OP1"]]: "OP",
                    self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["OP2"]]: "OP",
                    self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["O3'"]]: "OR",
                    self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["O52"]]: "OR",
                }
            )
        elif self.phosphate_style == "amber":
            pass
        else:
            raise RuntimeError(f"Unknown phosphate style {self.phosphate_style}")
    
        if self.alpha_gamma_style == "bsc0":
            modified_atom_types.update(
                {self.valid_fragment["fragment"].parent_fragment_mapping[self.mol_suger_symbol_index["C01"]]: "CI"}
            )
        elif self.alpha_gamma_style == "amber":
            pass
        else:
            raise RuntimeError(f"Unknown alpha gamma style {self.alpha_gamma_style}")
        
        parm7 = frag_dir/f"{self.mol_name}_frag_epsilon_zeta_{self.phosphate_style}P_{self.alpha_gamma_style}ag.parm7"
        self.parameterize(str(output_dir/"fragment_epsilon_zeta/fragment.pdb"), 
                          frag_dir, prefix=f"{self.mol_name}_frag_epsilon_zeta_{self.phosphate_style}P_{self.alpha_gamma_style}ag", 
                          addons=addons,  mod_atom_types=modified_atom_types)
        # parm7 = "/home/gridsan/ywang3/Project/Capping/test_parameterization/test_patch/CR2_frag_epsilon_zeta_cmap_linear.parm7"
        parm7 = "/home/gridsan/ywang3/Project/Capping/test_parameterization/test_patch/CR2_frag_epsilon_zeta_cmap_cubic.parm7"
        parm_mol = pmd.load_file(str(parm7))
        for idx, atom in enumerate(parm_mol.atoms):
            atom.charge = charged_pmd_mol.atoms[idx].charge
        logger.info(f'collecting QM and MM Energies for fragment Epsilon / Zeta')
        
        dihedrals = {}
        mm_energies = []
        qm_energies = []
        conf_names = []

        for conf in self.all_conformer_files:
            # >> read QM energy
            if self.aqueous:
                log_file = conf.parent/f"{conf.stem}_property.txt"
                qm_energy = read_energy_from_txt(log_file)
            else:
                log_file = conf.with_suffix(".psi4.log")
                qm_energy = read_energy_from_log(log_file)
            # >> calculate MM energy
            conf_mol = rd_load_file(str(conf))
            # unit from RDkit is Angstrom, convert to nm by dividing 10.
            positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
            
            parm_mol.positions = positions            
            mm_restraints = []
            dihedral_each_conf = {}

            dihedral_degree_epsilon = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["epsilon"])[0]].measure())
            dihedral_degree_zeta = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["zeta"])[0]].measure())
            dihedral_each_conf["epsilon"] = dihedral_degree_epsilon / 180.0 * np.pi
            dihedral_each_conf["zeta"] = dihedral_degree_zeta / 180.0 * np.pi
            mm_restraints.append({"atom_index": self.valid_fragment["epsilon"], "value": dihedral_degree_epsilon})
            mm_restraints.append({"atom_index": self.valid_fragment["zeta"], "value": dihedral_degree_zeta})
            mm_restraints.extend([x for x in self.valid_fragment["fragment_constraints"].values()])
            logger.info(conf.stem)
            
            mm_energy = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=self.aqueous,
                                                    optimize=True, 
                                                    restraints=mm_restraints,
                                                    save=False, 
                                                    output_file=f"{conf.stem}_mm_opmized.pdb")
                                                    # output_file=conf.parent/f"{conf.stem}_mm_optimized.pdb")
            dihedrals[conf.stem] = dihedral_each_conf
            mm_energies.append(mm_energy)
            qm_energies.append(qm_energy * Hatree2kCalPerMol)
            conf_names.append(conf.stem)

        mm_energy = np.array(mm_energies)
        qm_energy = np.array(qm_energies)
        mm_energy = mm_energy - mm_energy.min()
        qm_energy = qm_energy - qm_energy.min()
        self.conformer_data = {}
        for idx, conf_name in enumerate(conf_names):
            self.conformer_data[conf_name] = {
                    # "dihedral": float(dihedrals[idx]),
                    "dihedral": dihedrals[conf_name],
                    "mm_energy": float(mm_energy[idx]),
                    "qm_energy": float(qm_energy[idx])
                }        
        save_to_yaml(self.conformer_data, output_dir/f"conformer_cmap_epsilon_zeta_{self.phosphate_style}P_{self.alpha_gamma_style}ag{suffix}.yaml")
        

class SugarPuckerTorsionFactory(MoleculeFactory):
    def __init__(self, 
                order: int = 4, 
                panelty_weight: float = 0.1, 
                threshold: float = 0.01, 
                mol_name: str = "mol",
                atom_type: str = "amber",
                method: str = "wB97X-V",
                basis: str = "def2-TZVPD",
                threads: int = 48,
                memory: str = "160 GB",
                resp_method: str = "HF",
                resp_basis: str = "6-31+G*",
                resp_n_conformers: int = 6,
                output_dir: str = None,
                cap_methylation: bool = True,
                aqueous: bool = False,
                fix_phase: bool = True,
                determine_bond_orders: bool = False,
                pairwise: bool = False,
                constrain_chi: bool = True,
                pseudo_angle_grids = list(range(0, 180, 9)) + [180.]  # 21
        ):
       
        self.basis = basis
        self.method = method
        self.order = order
        self.panelty_weight = panelty_weight
        self.threshold = threshold
        self.threads = threads
        self.memory = memory
        self.resp_method = resp_method
        self.resp_basis = resp_basis
        self.resp_n_conformers = resp_n_conformers
        self.output_dir = output_dir
        self.cap_methylation = cap_methylation
        self.aqueous = aqueous
        self.fix_phase = fix_phase
        self.determine_bond_orders = determine_bond_orders
        self.pairwise = pairwise
        self.mol_name = mol_name
        self.atom_type = atom_type
        self.constrain_chi = constrain_chi
        self.pseudo_angle_grids = pseudo_angle_grids
        self.mol = None
        self.sugar_template_file = str(TEMPLATE_DIR/"sugar_template.pdb")
        self.sugar_template_mol = Chem.MolFromPDBFile(str(self.sugar_template_file), removeHs=False)
        
    def match_template(self):
        r"""Match the template molecule with the input molecule.
        Two template molecules are used:
            1. The nucleotide template molecule.
            2. The sugar template molecule.
        """
        self.sugar_fragment_mapping = map_atoms(self.sugar_template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        sugar_name_to_index = atomName_to_index(self.sugar_template_mol)
        mapping_dict = {}
        for atom in self.sugar_fragment_mapping:
            mapping_dict[atom[0]] = atom[1]
        self.sugar_fragment_mapping = mapping_dict
        _atom_names = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'","O4'", "O5'", "HO3'", "HO2'", "HO5'"]
        self.mol_suger_symbol_index = {}
        for atom_name in _atom_names:
            if sugar_name_to_index[atom_name] in self.sugar_fragment_mapping:
                self.mol_suger_symbol_index[atom_name] = self.sugar_fragment_mapping[sugar_name_to_index[atom_name]]
    
    def fragmentize(self, output_dir, extra_restraints=None):
        """Fragmentize the molecule by O3'-P bond.
        """
        self.match_template()
        SPTF = TorsionFragmentizer(
                self.mol, 
                cap_methylation=False, 
                determine_bond_orders=False,
                break_aromatic_ring=True
            )
        # find glycosidic bond by rotatable bond associated with C1' atom
        rotatable_bonds = SPTF.get_rotatable_bonds()
        C1p_rotatable_bonds = []
        for rbond in rotatable_bonds:
            if self.mol_suger_symbol_index["C1'"] in rbond:
                C1p_rotatable_bonds.append(rbond)
        if len(C1p_rotatable_bonds) == 0:
            raise RuntimeError("No rotatable bond associated with C1' atom.")
        elif len(C1p_rotatable_bonds) > 1:
            raise RuntimeError("More than one rotatable bond associated with C1' atom.")
        target_bond = C1p_rotatable_bonds[0]
        
        frag = SPTF.fragment_on_bond(
            target_bond[0],
            target_bond[1],
        )
        
        frag.save(str(output_dir/f"fragment_sugar_pucker"), format="pdb")
        logger.info(f"Sugar pucker Rotatable bond is valid for fragment ")
        logger.info(f"Fragment is saved to {str(output_dir/f'fragment_sugar_pucker')}")
        self.valid_fragment = {
            "fragment": frag,
            "glycosidic": frag.pivotal_dihedral_quartet[0],
            "sugar-v3": tuple(frag.parent_fragment_mapping[self.mol_suger_symbol_index[x]] for x in ["O4'", "C4'", "C3'", "C2'"]),
            "sugar-v1": tuple(frag.parent_fragment_mapping[self.mol_suger_symbol_index[x]] for x in ["O4'", "C1'", "C2'", "C3'"]),
            "fragment_constraints": {}
        }
        # DIHDEDRAL_CONSTRAINTS
        for ctype in ["O3'-constraint", "O2'-constraint", "O5'-constraint"]:
            constraints = DIHDEDRAL_CONSTRAINTS[ctype]
            atom_symbols = constraints["atoms"]
            if np.all([self.mol_suger_symbol_index[a] in frag.parent_fragment_mapping for a in atom_symbols]):
                self.valid_fragment["fragment_constraints"][ctype] = {
                    "type": "dihedral",
                    "atom_index": [frag.parent_fragment_mapping[self.mol_suger_symbol_index[a]] for a in atom_symbols],
                    "value": constraints["angle"],
                }
        if extra_restraints is not None:
            self.valid_fragment["fragment_constraints"]["extra"] = {
                "type": "dihedral",
                "atom_index": extra_restraints["index"], "value": extra_restraints["value"]
            }
    
    def gen(self, submit=False, local=False, overwrite=False, extra_restraints=None):
        output_dir = Path(self.output_dir)
        logger.info(f" >>> Output directory: {output_dir} <<<")
        self.fragmentize(output_dir, extra_restraints=extra_restraints)
        self.all_conformer_files = []
        self.slurm_files = []
        
        logger.info(f"Scanning dihedral Sugar Pucker")
        v1_list = []
        v3_list = []
        for pa in self.pseudo_angle_grids:
            v1, v3 = get_suger_picker_angles_from_pseudorotation(pa, PSEUDOROTATION["C3'-endo"]["intensity"])
            v1_list.append(v1)
            v3_list.append(v3)
        
        dsc = DihedralScanner(
            input_file=str(output_dir/f"fragment_sugar_pucker/fragment.pdb"), 
            dihedrals=[self.valid_fragment["sugar-v3"]],
            charge=self.valid_fragment["fragment"].charge,
            workdir=str(output_dir/f"fragment_sugar_pucker"),
            conformer_prefix=f"frag_conformer_sugar_pucker",
            constraints=[[x["atom_index"], x["value"]] for x in self.valid_fragment["fragment_constraints"].values()],
            force_constant=1.0,  # Hartree/(Bohr**2)
            warming_constraints=True
        ) 
        concurrent_constraints = []
        for idx, v1 in enumerate(v1_list):
            concurrent_constraints.append([[[x for x in self.valid_fragment["sugar-v1"]], v1],])
        dsc.run_on_grids(
            v3_list,
            concurrent_constraints=concurrent_constraints,
            overwrite=overwrite
        )
        self.all_conformer_files.extend(dsc.conformers[self.valid_fragment["sugar-v3"]])
    
        for idx, conf in enumerate(self.all_conformer_files):
            slurm_file = (output_dir/f"fragment_sugar_pucker/conf{idx}.slurm").resolve()
            if self.valid_fragment["fragment"].charge != 0:
                logger.warning(f"Fragment has charge {self.valid_fragment['fragment'].charge}")
                logger.warning(f"Please use appropriate QM methods for charged fragments")
            self.write_script(
                slurm_filename=slurm_file,
                input_file=str(conf),
                output_dir=(output_dir/f"fragment_sugar_pucker").resolve(),
                charge=self.valid_fragment["fragment"].charge,
                local=local,
                aqueous=self.aqueous
            )
            self.slurm_files.append(slurm_file)
            
            if not overwrite and ((os.path.exists(conf.with_suffix(".psi4.log")) and not self.aqueous) \
                or (os.path.exists(conf.with_suffix(".molden")) and self.aqueous)):
                continue
            
            cwd = Path.cwd().resolve()
            os.chdir(output_dir)
            if local:
                logger.info(f"Running {slurm_file}")
                os.system(f"bash {slurm_file}")
                os.remove(slurm_file)
            if submit:
                logger.info(f"Submiting {slurm_file}")
                os.system(f"LLsub {slurm_file}")
            for suffix in [".gbw", ".cpcm", ".densities"]:
                if os.path.exists(conf.with_suffix(suffix)):
                    os.remove(conf.with_suffix(suffix))
            os.chdir(cwd)
    
    def optimize(self, parameter_mod=None, overwrite=False, suffix=""):
        output_dir = Path(self.output_dir)
        logger.info(f"Optimizing fragment Sugar Pucker")
        logger.info(f"calculating atomic charges for fragment Sugar Pucker")
        frag_dir = Path(output_dir / f"fragment_sugar_pucker")
        # get atomic charges
        # >>>    CHARGE    <<<
        charged_mol2 = frag_dir/f"{self.all_conformer_files[0].stem}.resp.mol2"
        if overwrite or (not os.path.exists(charged_mol2)):
            RESP_fragment(
                select_from_list(self.all_conformer_files, self.resp_n_conformers, method="even"),
                self.valid_fragment["fragment"].charge,
                str(frag_dir),
                self.mol_name,
                memory=self.memory, 
                n_threads=self.threads, 
                method_basis=f"{self.resp_method}/{self.resp_basis}",
            )
        else:
            logger.info(f"File exised, using existing charge file {charged_mol2}.")
        charged_pmd_mol = pmd.load_file(str(charged_mol2))
        logger.info("Parameterizing fragment")
        parm7 = frag_dir/f"{self.mol_name}_frag_sugar_pucker.parm7"
        
        if self.aqueous:
            addons = ["set default PBRadii mbondi"]
        else:
            addons = []
        self.parameterize(str(output_dir/"fragment_sugar_pucker/fragment.pdb"), 
                          frag_dir, prefix=f"{self.mol_name}_frag_sugar_pucker", 
                          addons=addons,)
        
        parm_mol = pmd.load_file(str(parm7))
        for idx, atom in enumerate(parm_mol.atoms):
            atom.charge = charged_pmd_mol.atoms[idx].charge
        
        # START  >>> turn off the dihedral terms along the rotatable bond <<<
        idx_list_sugar_v1 = self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["sugar-v1"])
        idx_list_sugar_v3 = self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["sugar-v3"])
        assert len(idx_list_sugar_v1) > 0, f"No dihedral term found for fragment sugar-v1." 
        assert len(idx_list_sugar_v3) > 0, f"No dihedral term found for fragment sugar-v3."
        for idx_list in [idx_list_sugar_v1, idx_list_sugar_v3]:
            for dih_idx in idx_list:
                parm_mol.dihedrals[dih_idx].type.phi_k = 0.0
        if parameter_mod is not None:
            parm_mol = modify_torsion_parameters(parm_mol, parameter_mod)
        
        logger.info(f'collecting QM and MM Energies for fragment Sugar Pucker')
        
        dihedrals = {}
        mm_energies = []
        qm_energies = []
        conf_names = []

        for conf in self.all_conformer_files:
            # >> read QM energy
            if self.aqueous:
                log_file = conf.parent/f"{conf.stem}_property.txt"
                qm_energy = read_energy_from_txt(log_file)
            else:
                log_file = conf.with_suffix(".psi4.log")
                qm_energy = read_energy_from_log(log_file)
            # >> calculate MM energy
            conf_mol = rd_load_file(str(conf))
            # unit from RDkit is Angstrom, convert to nm by dividing 10.
            positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
            parm_mol.positions = positions
            # >> NEW VERSION [ALL dihedral terms were considered]
            
            mm_restraints = []
            dihedral_each_conf = {}

            dihedral_v1 = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["sugar-v1"])[0]].measure())
            dihedral_v3 = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, self.valid_fragment["sugar-v3"])[0]].measure())

            dihedral_each_conf["sugar-v1"] = dihedral_v1 / 180.0 * np.pi
            dihedral_each_conf["sugar-v3"] = dihedral_v3 / 180.0 * np.pi
            mm_restraints.append({"atom_index": self.valid_fragment["sugar-v1"], "value": dihedral_v1})
            mm_restraints.append({"atom_index": self.valid_fragment["sugar-v3"], "value": dihedral_v3})
            
            mm_restraints.extend([x for x in self.valid_fragment["fragment_constraints"].values()])
            
            mm_energy = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=self.aqueous,
                                                    optimize=True, 
                                                    restraints=mm_restraints,
                                                    save=False, output_file=conf.parent/f"{conf.stem}_mm_optimized.pdb")
            dihedrals[conf.stem] = dihedral_each_conf
            mm_energies.append(mm_energy)
            qm_energies.append(qm_energy * Hatree2kCalPerMol)
            conf_names.append(conf.stem)

        mm_energy = np.array(mm_energies)
        qm_energy = np.array(qm_energies)
        mm_energy = mm_energy - mm_energy.min()
        qm_energy = qm_energy - qm_energy.min()
        self.conformer_data = {}
        for idx, conf_name in enumerate(conf_names):
            self.conformer_data[conf_name] = {
                    "dihedral": dihedrals[conf_name],
                    "mm_energy": float(mm_energy[idx]),
                    "qm_energy": float(qm_energy[idx])
                }
        optimizer = TorsionOptimizer(self.order, self.panelty_weight, self.threshold, fix_phase=self.fix_phase, n_dihedrals=2)
        # of shape [n_conformers, n_dihedrals]
        dihedral_value_matrix = np.array([[dihedrals[conf_name][x] for x in ["sugar-v1", "sugar-v3"]] for conf_name in conf_names])
        
        optimizer.infer_parameters(
            dihedrals=dihedral_value_matrix,
            energy_mm=mm_energy,
            energy_qm=qm_energy,
            pairwise=self.pairwise
        )
        self.parameter_set = optimizer.get_parameters()
        self.parameter_set["dihedral"] = ["sugar-v1", "sugar-v3"]
        
        save_to_yaml(self.conformer_data, output_dir/f"conformer_sugar_pucker{suffix}.yaml")
        save_to_yaml(self.parameter_set, output_dir/f"parameter_sugar_pucker{suffix}.yaml")
        

class ChiTorsionFactory(MoleculeFactory):
    def __init__(self, 
                order: int = 4, 
                panelty_weight: float = 0.1, 
                threshold: float = 0.01, 
                mol_name: str = "mol",
                atom_type: str = "amber",
                method: str = "wB97X-V",
                basis: str = "def2-TZVPD",
                threads: int = 48,
                memory: str = "160 GB",
                resp_method: str = "HF",
                resp_basis: str = "6-31+G*",
                resp_n_conformers: int = 6,
                output_dir: str = None,
                cap_methylation: bool = True,
                aqueous: bool = False,
                fix_phase: bool = True,
                determine_bond_orders: bool = False,
                pairwise: bool = False,
                constrain_chi: bool = True,
                chi_angle_grids = list(range(0, 360, 10))  # 36
        ):
        self.basis = basis
        self.method = method
        self.order = order
        self.panelty_weight = panelty_weight
        self.threshold = threshold
        self.threads = threads
        self.memory = memory
        self.resp_method = resp_method
        self.resp_basis = resp_basis
        self.resp_n_conformers = resp_n_conformers
        self.output_dir = output_dir
        self.cap_methylation = cap_methylation
        self.aqueous = aqueous
        self.fix_phase = fix_phase
        self.determine_bond_orders = determine_bond_orders
        self.pairwise = pairwise
        self.mol_name = mol_name
        self.atom_type = atom_type
        self.constrain_chi = constrain_chi
        self.chi_angle_grids = chi_angle_grids
        self.mol = None
        self.sugar_template_file = str(TEMPLATE_DIR/"sugar_template.pdb")
        self.sugar_template_mol = Chem.MolFromPDBFile(str(self.sugar_template_file), removeHs=False)
        
    def match_template(self):
        r"""Match the template molecule with the input molecule.
        Two template molecules are used:
            1. The nucleotide template molecule.
            2. The sugar template molecule.
        """
        self.sugar_fragment_mapping = map_atoms(self.sugar_template_mol, self.mol, ringMatchesRingOnly=False, completeRingsOnly=False)
        sugar_name_to_index = atomName_to_index(self.sugar_template_mol)
        mapping_dict = {}
        for atom in self.sugar_fragment_mapping:
            mapping_dict[atom[0]] = atom[1]
        self.sugar_fragment_mapping = mapping_dict
        _atom_names = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'","O4'", "O5'", "HO3'", "HO2'", "HO5'"]
        self.mol_suger_symbol_index = {}
        for atom_name in _atom_names:
            if sugar_name_to_index[atom_name] in self.sugar_fragment_mapping:
                self.mol_suger_symbol_index[atom_name] = self.sugar_fragment_mapping[sugar_name_to_index[atom_name]]
    
    def fragmentize(self, output_dir):
        """Fragmentize the molecule by O3'-P bond.
        """
        self.match_template()
        CTF = TorsionFragmentizer(
                self.mol, 
                cap_methylation=False, 
                determine_bond_orders=False,
                break_aromatic_ring=True
            )
        # find glycosidic bond by rotatable bond associated with C1' atom
        rotatable_bonds = CTF.get_rotatable_bonds()
        C1p_rotatable_bonds = []
        for rbond in rotatable_bonds:
            if self.mol_suger_symbol_index["C1'"] in rbond:
                C1p_rotatable_bonds.append(rbond)
        if len(C1p_rotatable_bonds) == 0:
            raise RuntimeError("No rotatable bond associated with C1' atom.")
        elif len(C1p_rotatable_bonds) > 1:
            raise RuntimeError("More than one rotatable bond associated with C1' atom.")
        target_bond = C1p_rotatable_bonds[0]
        
        frag = CTF.fragment_on_bond(
            target_bond[0],
            target_bond[1],
        )
        frag_dir = output_dir/"fragment_chi"
        frag.save(str(frag_dir), format="pdb")
        logger.info(f"Sugar pucker Rotatable bond is valid for fragment ")
        logger.info(f"Fragment is saved to {str(frag_dir)}")
        self.valid_fragment = {
            "fragment": frag,
            "glycosidic": frag.pivotal_dihedral_quartet[0],
            "sugar-v3": tuple(frag.parent_fragment_mapping[self.mol_suger_symbol_index[x]] for x in ["O4'", "C4'", "C3'", "C2'"]),
            "sugar-v1": tuple(frag.parent_fragment_mapping[self.mol_suger_symbol_index[x]] for x in ["O4'", "C1'", "C2'", "C3'"]),
            "fragment_constraints": {}
        }
        # DIHDEDRAL_CONSTRAINTS
        for ctype in ["O3'-constraint", "O2'-constraint", "C3'-endo-constraint-v1", "C3'-endo-constraint-v3"]:
            constraints = DIHDEDRAL_CONSTRAINTS[ctype]
            atom_symbols = constraints["atoms"]
            if np.all([self.mol_suger_symbol_index[a] in frag.parent_fragment_mapping for a in atom_symbols]):
                self.valid_fragment["fragment_constraints"][ctype] = {
                    "type": "dihedral",
                    "atom_index": [frag.parent_fragment_mapping[self.mol_suger_symbol_index[a]] for a in atom_symbols],
                    "value": constraints["angle"],
                }
    
    def gen(self, submit=False, local=False, overwrite=False):
        output_dir = Path(self.output_dir)
        logger.info(f" >>> Output directory: {output_dir} <<<")
        self.fragmentize(output_dir)
        self.all_conformer_files = []
        self.slurm_files = []
        
        logger.info(f"Scanning dihedral Chi")
        
        dsc = DihedralScanner(
            input_file=str(output_dir/f"fragment_chi/fragment.pdb"), 
            dihedrals=[self.valid_fragment["glycosidic"]],
            charge=self.valid_fragment["fragment"].charge,
            workdir=str(output_dir/f"fragment_chi"),
            conformer_prefix=f"frag_conformer_chi",
            constraints=[[x["atom_index"], x["value"]] for x in self.valid_fragment["fragment_constraints"].values()],
            force_constant=1.0,  # Hartree/(Bohr**2)
            warming_constraints=True
        ) 
        dsc.run_on_grids(
            self.chi_angle_grids,
            overwrite=overwrite
        )
        self.all_conformer_files.extend(dsc.conformers[self.valid_fragment["glycosidic"]])
    
        for idx, conf in enumerate(self.all_conformer_files):
            slurm_file = (output_dir/f"fragment_chi/conf{idx}.slurm").resolve()
            if self.valid_fragment["fragment"].charge != 0:
                logger.warning(f"Fragment has charge {self.valid_fragment['fragment'].charge}")
                logger.warning(f"Please use appropriate QM methods for charged fragments")
            self.write_script(
                slurm_filename=slurm_file,
                input_file=str(conf),
                output_dir=(output_dir/f"fragment_chi").resolve(),
                charge=self.valid_fragment["fragment"].charge,
                local=local,
                aqueous=self.aqueous
            )
            self.slurm_files.append(slurm_file)
            
            if not overwrite and ((os.path.exists(conf.with_suffix(".psi4.log")) and not self.aqueous) \
                or (os.path.exists(conf.with_suffix(".molden")) and self.aqueous)):
                continue
            
            cwd = Path.cwd().resolve()
            os.chdir(output_dir)
            if local:
                logger.info(f"Running {slurm_file}")
                os.system(f"bash {slurm_file}")
                os.remove(slurm_file)
            if submit:
                logger.info(f"Submiting {slurm_file}")
                os.system(f"LLsub {slurm_file}")
            for suffix in [".gbw", ".cpcm", ".densities"]:
                if os.path.exists(conf.with_suffix(suffix)):
                    os.remove(conf.with_suffix(suffix))
            os.chdir(cwd)
    
    def optimize(self, parameter_mod=None, overwrite=False, seed=1106, suffix=""):
        """Optimize the torsion parameters for the molecule.
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
        logger.info(f"Optimizing fragment Chi")
        logger.info(f"calculating atomic charges for fragment Chi")
        frag_dir = Path(output_dir/f"fragment_chi")
        # get atomic charges
        charged_mol2 = frag_dir/f"{self.all_conformer_files[0].stem}.resp.mol2"
        if overwrite or (not os.path.exists(charged_mol2)):
            self.charge_molecule(
                select_from_list(self.all_conformer_files, self.resp_n_conformers, method="even"),
                self.valid_fragment["fragment"].charge, frag_dir
            )
        else:
            logger.info(f"File exised, using existing charge file {charged_mol2}.")
        charged_pmd_mol = pmd.load_file(str(charged_mol2))
        logger.info("Parameterizing fragment")
        parm7 = frag_dir/f"{self.mol_name}_frag_chi.parm7"
        if overwrite or (not os.path.exists(parm7)):
            if self.aqueous:
                addons = ["set default PBRadii mbondi"]
            else:
                addons = []
            self.parameterize(str(output_dir/f"fragment_chi/fragment.pdb"), frag_dir, prefix=f"{self.mol_name}_frag_chi", addons=addons)
        else:
            logger.info(f"File exised, using existing parameter file {parm7}.")
        parm_mol = pmd.load_file(str(parm7))
        for idx, atom in enumerate(parm_mol.atoms):
            atom.charge = charged_pmd_mol.atoms[idx].charge

        # START  >>> turn off the dihedral terms along the rotatable bond <<<
        # idx_list_dihedral = self.get_dihrdeal_terms_by_quartet(parm_mol, vfrag["fragment"].pivotal_dihedral_quartet[0])
        idx_list_bond = self.get_dihedral_terms_by_bond(parm_mol, [self.valid_fragment["glycosidic"][1], self.valid_fragment["glycosidic"][2]])
        logger.info(f"Found {len(idx_list_bond)} dihedral terms by bonds: {idx_list_bond}")
        for dih_idx in idx_list_bond:
            parm_mol.dihedrals[dih_idx].type.phi_k = 0.0
        # END    >>> turn off the dihedral terms along the rotatable bond <<<
        if parameter_mod is not None:
            parm_mol = modify_torsion_parameters(parm_mol, parameter_mod)
            # import code; code.interact(local=locals())
        
        logger.info(f'collecting QM and MM Energies for fragment Chi')
        dihedrals = {}
        mm_energies = []
        qm_energies = []
        conf_names = []

        for conf in self.all_conformer_files:
            # >> read QM energy
            if self.aqueous:
                log_file = conf.parent/f"{conf.stem}_property.txt"
                qm_energy = read_energy_from_txt(log_file)
            else:
                log_file = conf.with_suffix(".psi4.log")
                qm_energy = read_energy_from_log(log_file)
            # >> calculate MM energy
            conf_mol = rd_load_file(str(conf))
            # unit from RDkit is Angstrom, convert to nm by dividing 10.
            positions = np.array(conf_mol.GetConformer().GetPositions()) / 10.
            parm_mol.positions = positions
            # >> NEW VERSION [ALL dihedral terms were considered]
            
            mm_restraints = []
            dihedral_each_conf = {}

            for dihedral_atom_index in self.valid_fragment["fragment"].all_dihedral_quartets:
                dihedral_degrees = float(parm_mol.dihedrals[self.get_dihrdeal_terms_by_quartet(parm_mol, dihedral_atom_index)[0]].measure())
                dihedral_each_conf[dihedral_atom_index] = dihedral_degrees / 180.0 * np.pi
                mm_restraints.append({"atom_index": dihedral_atom_index, "value": dihedral_degrees})
            mm_restraints.extend([x for x in self.valid_fragment["fragment_constraints"].values()])
            # >> NEW VERSION END
            
            # >> OLD VERSION [ONLY one dihedral term was considered]
            # dihedral_degrees = float(parm_mol.dihedrals[idx_list_dihedral[0]].measure())
            # dihedrals.append(dihedral_degrees / 180.0 * np.pi)
            # mm_restraints = [{"atom_index": vfrag["fragment"].pivotal_dihedral_quartet[0], "value": dihedral_degrees}]
            # mm_restraints.extend([x for x in vfrag["fragment_constraints"].values()])
            # >> OLD VERSION END
            
            mm_energy = self.calculate_mm_energy(parm_mol, positions, implicit_solvent=self.aqueous,
                                                    optimize=True, 
                                                    restraints=mm_restraints,
                                                    save=False, output_file=conf.parent/f"{conf.stem}_mm_optimized.pdb")
            dihedrals[conf.stem] = dihedral_each_conf
            mm_energies.append(mm_energy)
            qm_energies.append(qm_energy * Hatree2kCalPerMol)
            conf_names.append(conf.stem)

        mm_energy = np.array(mm_energies)
        qm_energy = np.array(qm_energies)
        mm_energy = mm_energy - mm_energy.min()
        qm_energy = qm_energy - qm_energy.min()
        for idx, conf_name in enumerate(conf_names):
            self.conformer_data[conf_name] = {
                    # "dihedral": float(dihedrals[idx]),
                    "dihedral": dihedrals[conf_name],
                    "mm_energy": float(mm_energy[idx]),
                    "qm_energy": float(qm_energy[idx])
                }
        optimizer = TorsionOptimizer(self.order, self.panelty_weight, self.threshold, fix_phase=self.fix_phase, 
                                     n_dihedrals=len(self.valid_fragment["fragment"].all_dihedral_quartets), seed=seed)
        # of shape [n_conformers, n_dihedrals]
        dihedral_value_matrix = np.array([[dihedrals[conf_name][x] for x in self.valid_fragment["fragment"].all_dihedral_quartets] for conf_name in conf_names])
        
        optimizer.infer_parameters(
            dihedrals=dihedral_value_matrix,
            energy_mm=mm_energy,
            energy_qm=qm_energy,
            pairwise=self.pairwise
        )
        self.parameter_set = optimizer.get_parameters()
        self.parameter_set["dihedral"] = [[self.valid_fragment["fragment"].fragment_parent_mapping[xi] for xi in x] for x in self.valid_fragment["fragment"].all_dihedral_quartets]

        save_to_yaml(self.conformer_data, output_dir/f"conformer_chi{suffix}.yaml")
        save_to_yaml(self.parameter_set, output_dir/f"parameter_chi{suffix}.yaml")
    
            