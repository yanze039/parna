import numpy as np
import rdkit
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import Conformer
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
# BUG: UFF can't recognize high-valanced P and S atoms. 
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField, MMFFGetMoleculeForceField, MMFFGetMoleculeProperties
# BUG: only when this import exists, UFFGetMoleculeForceField can be registered.
from rdkit.Chem import ChemicalForceFields

import os
import yaml
from parna.logger import getLogger
from parna.utils import map_atoms, normalize, rd_load_file
from pathlib import Path
from parna.parm import parameterize, generate_frcmod
from parna.resp import RESP_fragment
from parna.utils import atomName_to_index
from parna.utils import map_atoms, SLURM_HEADER_CPU

import parmed as pmd
from parmed.topologyobjects import DihedralType, Dihedral
from typing import Union, List

sugar_template_file = os.path.join(os.path.dirname(__file__), "template", "sugar_template.pdb")

# Define custom constructor for tuples
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

# Define custom representer for tuples
def tuple_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:python/tuple', data)

# Add the constructors to PyYAML
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
yaml.add_representer(tuple, tuple_representer)


logger = getLogger(__name__)


def flexible_align(mobile, template, atom_mapping=None, force_constant=100.0, max_iterations=6):
    if atom_mapping is None:
        atom_mapping = map_atoms(mobile, template)
    _ = rdMolAlign.AlignMol(mobile, template, atomMap=atom_mapping)
    mp = MMFFGetMoleculeProperties(mobile)
    ff = MMFFGetMoleculeForceField(mobile, mp)
    template_conf = template.GetConformer(0)
    for i_q, i_t in atom_mapping:
        p_t = template_conf.GetAtomPosition(i_t)
        pIdx = ff.AddExtraPoint(p_t.x, p_t.y, p_t.z, fixed=True) -1
        ff.MMFFAddDistanceConstraint(pIdx, i_q, False, 0., 0., force_constant)
    ff.Initialize()
    logger.info("Constrained Energy minimization...")
    for it in range(max_iterations):
        logger.info(f"Minimization iteration {it+1}")
        more = ff.Minimize(maxIts=20000, energyTol=1e-4, forceTol=1e-3)
        if int(more) == 0:
            logger.info("Converged!")
            break

    return mobile


def get_bond_vector(mol, index1, index2):
    pos1 = mol.GetConformer().GetAtomPosition(index1)
    pos2 = mol.GetConformer().GetAtomPosition(index2)
    return np.array(pos2) - np.array(pos1)


def get_norm_vector(mol, center, ter1, ter2):
    bvec1 = get_bond_vector(mol, center, ter1)
    bvec2 = get_bond_vector(mol, center, ter2)
    norm_bvec = np.cross(bvec1, bvec2)
    norm_bvec = norm_bvec / np.linalg.norm(norm_bvec)
    return norm_bvec


def ortho_frame(vec1, vec2):
    x = normalize(vec1)
    y = normalize(np.cross(vec1, vec2))
    z = normalize(np.cross(x, y))
    return np.stack([x,y,z])


def construct_local_frame(mol, center, ringatom1, ringatom2, sugeratom):
    """Construct the local frame of the base. represented by a triangle in 3d space.
        Args:
            center: atom index of the nitrogen center;
            ringatom1: atom index of first atom on the base ring;
            ringatom2: atom index of second atom on the base ring;
            sugeratom: atom index of the C1' on suger.
    
    """
    posc = mol.GetConformer().GetAtomPosition(center)
    pos1 = mol.GetConformer().GetAtomPosition(ringatom1)
    pos2 = mol.GetConformer().GetAtomPosition(ringatom2)
    poss = mol.GetConformer().GetAtomPosition(sugeratom)
    # one edge is just the bond pointing from nitrogen to suger
    vector1 = normalize(np.array(poss) - np.array(posc))
    # second edge is the normal of the plane
    bvec1 = np.array(pos1) - np.array(posc)
    bvec2 = np.array(pos2) - np.array(posc)
    norm_bvec = np.cross(bvec1, bvec2)
    norm_bvec = norm_bvec / np.linalg.norm(norm_bvec)
    return vector1, norm_bvec


def rotate_conformer(conformer: Conformer, matrix: np.ndarray):
    """
    Apply a 3x3 transformational matrix to the coordinates of the conformer.

    This is a convenience function that fits the 3x3 matrix into a 4x4 matrix to
    include the extra translational dimension required by `TransformConformer`.

    :param conformer: a conformer with nonzero coordinates
    :param matrix: a 3x3 matrix to transform the conformer coordinates
    """
    mat_with_translation = np.identity(4)
    for idx, col in enumerate(matrix):
        mat_with_translation[idx][:3] = col
    rdMolTransforms.TransformConformer(conformer, mat_with_translation)


def translate_conformer(conformer: Conformer, matrix: np.ndarray):
    """
    Apply a 3x3 transformational matrix to the coordinates of the conformer.

    This is a convenience function that fits the 3x3 matrix into a 4x4 matrix to
    include the extra translational dimension required by `TransformConformer`.

    :param conformer: a conformer with nonzero coordinates
    :param matrix: a 3x3 matrix to transform the conformer coordinates
    """
    mat_with_translation = np.identity(4)
    mat_with_translation[:3, 3] = matrix
    rdMolTransforms.TransformConformer(conformer, mat_with_translation)


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
    
    def charge_molecule(self, 
                        input_files: Union[str, List[str]],
                        charge, 
                        output_dir):
        """
        Calculate RESP charges for a molecule.
        Input:
            input_file: str, `PDB` or `XYZ` file. or a list of `PDB` or `XYZ` files
            charge: int, charge of the molecule
            output_dir: str, path to the output directory
        """
        RESP_fragment(
            input_files,
            int(charge),
            str(output_dir),
            self.mol_name,
            memory=self.memory, 
            n_threads=self.threads, 
            method_basis=f"{self.resp_method}/{self.resp_basis}",
        )
    
    def parameterize(self, pdb_file, output_dir, 
                     prefix, addons=[], mod_atom_types=None):
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
            "-pf", "y",
            "-rn", self.mol_name,
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
        if mod_atom_types is not None:
            for atom_idx, atom_type in mod_atom_types.items():
                mol2_pmd.atoms[atom_idx].type = atom_type
        tmp_lib = (output_dir/f"{pdb_file.stem}.tmp.lib")
        tmp_pdb = (output_dir/f"{pdb_file.stem}.tmp.pdb")
        mol2_pmd.save(str(tmp_lib), overwrite=True)
        mol2_pmd.save(str(tmp_pdb), overwrite=True)
        additional_frcmods = [
            str(output_dir/f"{pdb_file.stem}.frcmod"),
            str(Path(__file__).parent / "data" / "frcmod.99bsc0-chiol3-CaseP.frcmod")
        ]
        parameterize(
            oligoFile=str(tmp_pdb),
            external_libs=str(tmp_lib), 
            additional_frcmods=additional_frcmods,
            output_dir=output_dir,
            prefix=prefix,
            solvated=False,
            saveparm=True,
            check_atomtypes=False,
            addons=addons
        )
        for tmp_file in [tmp_mol2, tmp_lib, tmp_pdb]:
            if tmp_file.exists():
                os.remove(tmp_file)
    
    @staticmethod
    def calculate_mm_energy(parm, positions: np.ndarray, implicit_solvent=False, sampling=True,
                            optimize=False, restraints=None, force_constant=200., save=False, 
                            temperature=500., output_file=None, threads=40):
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        logger.info(f"Calculating energy for {parm}")
        if not implicit_solvent:
            logger.info("No implicit solvent")
            system = parm.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        else:
            logger.info("Using implicit solvent")
            system = parm.createSystem(
                nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                implicitSolvent=app.HCT) # corresponding to igb=1 in Amber)
        properties={}
        minimization_platform = mm.Platform.getPlatformByName('CPU')
        properties["Threads"]=str(threads)       
        integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, 1.0*unit.femtosecond)
        integrator.setRandomNumberSeed(1106)
        simulation = app.Simulation(parm.topology, system, integrator, minimization_platform, properties)
        context = simulation.context
        context.setPositions(positions*unit.nanometers)
        context.setVelocitiesToTemperature(temperature*unit.kelvin, 1106)
        for f in system.getForces():
            f.setForceGroup(0)
        if restraints is not None:
            restraint_force = mm.PeriodicTorsionForce()
            system.addForce(restraint_force)
            for res in restraints:
                logger.info(f"Adding torsion restraint between {res['atom_index']} at {res['value']} degrees")
                restraint_force.addTorsion(
                    res["atom_index"][0], res["atom_index"][1], res["atom_index"][2], res["atom_index"][3],
                    1, (res["value"]+180.)*unit.degrees, force_constant*unit.kilojoules_per_mole 
                )
            restraint_force.setForceGroup(1)
            context.reinitialize(preserveState=True)
            restraint_force.updateParametersInContext(context)
            
            # use CustomTorsionForce to add restraints (Harmonic potential)
            # force = mm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535")
            # force.addGlobalParameter("k", force_constant*unit.kilojoules_per_mole/unit.radians**2)
            # force.addPerTorsionParameter("theta0")
            # for res in restraints:
            #     logger.info(f"Adding torsion restraint between {res['atom_index']} at {res['value']} degrees")
            #     force.addTorsion(*res["atom_index"], [res["value"]*np.pi/180.])
            # system.addForce(force)
            # force.setForceGroup(1)
            # context.reinitialize(preserveState=True)
            # force.updateParametersInContext(context)
            
        if optimize:
            logger.info("Minimizing the structure...")
            if restraints is not None:
                for force in system.getForces():
                    if force.getForceGroup() == 1:
                        for di in range(force.getNumTorsions()):
                            atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(di)
                            new_k = force_constant*unit.kilojoules_per_mole
                            force.setTorsionParameters(di, atom1, atom2, atom3, atom4, periodicity, phase, new_k)
                            simulation.context.reinitialize(preserveState=True)
                            force.updateParametersInContext(simulation.context)
            if sampling:
                simulation.step(2000)
            simulation.minimizeEnergy()
        state = simulation.context.getState(getEnergy=True, getPositions=True, getForces=True, groups={0})
        energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilocalories_per_mole)
        if save:
            # save the optimized structure
            if output_file is None:
                output_file = "optimized.pdb"
            with open(output_file, "w") as f:
                app.PDBFile.writeFile(parm.topology, state.getPositions(), f)
        return energy
    
    @staticmethod
    def calculate_mm_energy_sander(parm7: str, rst7: str, workdir="."):
        parm7 = Path(parm7)
        mol_name = parm7.stem
        workdir = Path(workdir)
        cwd = os.getcwd()
        cpptraj_content = f"""parm {parm7} 
        trajin {rst7} 
        esander MOL out {mol_name}.dat gbsa 0 igb 
        """
        with open(workdir/f"{mol_name}.cpptraj.in", "w") as f:
            f.write(cpptraj_content)
        os.chdir(workdir)
        os.system("cpptraj -i cpptraj.in")
        os.chdir(cwd)
    
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
    
    def write_script_aq(self, slurm_filename, input_file, output_dir, charge, local=False):
        if local:
            slurm_content = "#!/bin/bash\n"
        else:
            slurm_content = SLURM_HEADER_CPU
        orca_python_exec = Path(__file__).parent / "qm" / "orca_calculate_energy.py"
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
        psi4_python_exec = Path(__file__).parent / "qm" / "psi4_calculate_energy.py"
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
    
    def get_dihedral_terms_by_bond(self, parm_mol, atom_index_list):
        """Get the dihedral terms by the bond indices.
        
        Input:
            parm_mol: parmed.Structure
            atom_index_list: list, the indices of the atoms of the rotatable bond.
        """
        idx_list = []
        atom_index_list = np.array(sorted(atom_index_list))
        for idx, dihedral in enumerate(parm_mol.dihedrals):
            _atom_index_list = np.array(sorted([dihedral.atom2.idx, dihedral.atom3.idx]))
            if np.all(_atom_index_list == atom_index_list):
                idx_list.append(idx)
        return idx_list
    
    def get_dihrdeal_terms_by_quartet(self, parm_mol, atom_index_list):
        """Get the dihedral terms by the bond indices.
        
        Input:
            parm_mol: parmed.Structure
            atom_index_list: list, the indices of the atoms of the diheral.
        """
        idx_list = []
        atom_index_list = np.array(sorted(atom_index_list))
        for idx, dihedral in enumerate(parm_mol.dihedrals):
            _atom_index_list = np.array(sorted([dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]))
            if np.all(_atom_index_list == atom_index_list):
                idx_list.append(idx)
        return idx_list
    
    def load_yaml(self, yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data
        
    
class SugarPatcher:
    def __init__(self, mol, template_file=sugar_template_file):
        self.mol = mol
        tmpl_mol = Chem.MolFromPDBFile(template_file, removeHs=False)
        self.atom_mapping = map_atoms(tmpl_mol, self.mol)
        self.tmpl_n2i = atomName_to_index(tmpl_mol)
        self.tmpl2query = {k: v for k, v in self.atom_mapping}
        
    def get_index_by_canonical_name(self, name):
        return self.tmpl2query[self.tmpl_n2i[name]]
        
    def patch_three_prime_end(self, group_smile='[P](=[O])([O-])(OC)', seed=1106):
        rdmol = copy.deepcopy(self.mol)
        query_o3p = self.tmpl2query[self.tmpl_n2i["O3'"]]
        query_ho3 = self.tmpl2query[self.tmpl_n2i["HO3'"]]
        atom_idx = query_ho3 
        connecting_atom_idx = query_o3p
        fragment = self.patch_group(rdmol, atom_idx, connecting_atom_idx, group_smile=group_smile, seed=seed)
        return fragment
    
    def patch_five_prime_end(self, group_smile='[P](=[O])([O-])(OC)', seed=1106):
        rdmol = copy.deepcopy(self.mol)
        query_o5p = self.tmpl2query[self.tmpl_n2i["O5'"]]
        query_ho5 = self.tmpl2query[self.tmpl_n2i["HO5'"]]
        atom_idx = query_ho5
        connecting_atom_idx = query_o5p
        fragment = self.patch_group(rdmol, atom_idx, connecting_atom_idx, group_smile=group_smile, seed=seed)
        return fragment
        
    @staticmethod
    def patch_group(rdmol, atom_idx, connecting_atom_idx=None, group_smile='[P](=[O])([O-])(OC)', seed=1106):
        
        if connecting_atom_idx is None:
            breaking_bond = rdmol.GetAtomWithIdx(atom_idx).GetBonds()
            assert len(breaking_bond) == 1
            breaking_bond = breaking_bond[0]
            connecting_atom_idx = breaking_bond.GetOtherAtomIdx(atom_idx)
        else:
            breaking_bond = rdmol.GetBondBetweenAtoms(atom_idx, connecting_atom_idx)
        
        frag_on3p = Chem.FragmentOnBonds(
                            rdmol, \
                            [breaking_bond.GetIdx()], \
                            addDummies=True, \
                            bondTypes=[breaking_bond.GetBondType()])
        frags_assigned = []
        frags_mol_atom_mapping = []
    
        frags_mols = Chem.GetMolFrags(
            frag_on3p, asMols=True, sanitizeFrags=True, 
            frags=frags_assigned, fragsMolAtomMapping=frags_mol_atom_mapping)
        desried_frag = frags_mols[frags_assigned[connecting_atom_idx]]
        P_patch = Chem.MolFromSmiles(group_smile)
        P_patch = Chem.AddHs(P_patch)
        for atom in P_patch.GetAtoms():
            atom.SetProp("_PatchAtom", "1")
        
        for atom in desried_frag.GetAtoms():
            atom.SetProp("_PatchAtom", "0")
                
        fragment = Chem.ReplaceSubstructs(desried_frag, 
                            Chem.MolFromSmiles('*'), 
                            P_patch,
                            replaceAll=True)[0]
        AllChem.EmbedMolecule(fragment, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(fragment)
        return fragment
    
    def patch_both_end(self):
        raise NotImplementedError("This method is not implemented yet.")
    
    def save_to_pdb(self, mol, filename):
        Chem.MolToPDBFile(mol, filename)


def modify_torsion_parameters(pmd_mol, dih_parm, resid=None):
    r"""Modify the torsion parameters of the molecule (`parmed.Structure`).
    
    Args:
        `pmd_mol`: parmed.Structure, the molecule to be modified.
        `new_parameter_set`: dict, the new parameter set.
    
    Returns:
        `pmd_mol`: `parmed.Structure`. the modified molecule.
    
    Example:
        ```
        dih_parm = {
            "dihedral": [[0, 1, 2, 3], [1, 2, 3, 4]],
            "k": [[0.1, 0.2], [0.3, 0.4]],
            "phase": [[0.0, pi], [0.0, pi]],
            "periodicity": [[1, 1], [1, 1]]
        }
        ```
    """
    
    dihedrals = dih_parm["dihedral"]
    ks = dih_parm["k"]
    phases = dih_parm["phase"]
    periodicities = dih_parm["periodicity"]
    assert len(dihedrals) == len(ks) == len(phases) == len(periodicities), "The length of dihedral, k, phase, and periodicity should be the same."
    for i, atom_idx in enumerate(dihedrals):
        dihedral_list = []
        for idx, dihedral in enumerate(pmd_mol.dihedrals):
            if resid is not None:
                if dihedral.atom1.residue.idx != resid:
                    continue
            _atom_idx = [dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]
            if np.all(np.array(sorted(atom_idx)) == np.array(sorted(_atom_idx))):
                dihedral_list.append(idx)
        dihedral_list.sort(reverse=True)
        assert len(dihedral_list) > 0, f"no dihedral found for {dihedral}"
        logger.info(f"Modifying dihedral {atom_idx}")
        for idx in dihedral_list:
            pmd_mol.dihedrals[idx].delete()
            del pmd_mol.dihedrals[idx]        
            
        for pi in range(len(ks[i])):  
            phase_angle = phases[i][pi]
            while phase_angle < -np.pi:
                phase_angle += 2*np.pi  
            while phase_angle > np.pi:
                phase_angle -= 2*np.pi    
            new_dih_typ = DihedralType(
                phi_k=ks[i][pi] ,
                per=periodicities[i][pi],
                phase=phase_angle / np.pi * 180.0,
                scee=1.200,
                scnb=2.000
            )
            
            exists = False
            # Do not add a duplicate dihedral type
            for dih_typ in pmd_mol.dihedral_types:
                if new_dih_typ == dih_typ:
                    new_dih_typ = dih_typ
                    exists = True
                    break

            if not exists:
                pmd_mol.dihedral_types.append(new_dih_typ)
                new_dih_typ.list = pmd_mol.dihedral_types
            
            atm1 = pmd_mol.atoms[atom_idx[0]]
            atm2 = pmd_mol.atoms[atom_idx[1]]
            atm3 = pmd_mol.atoms[atom_idx[2]]
            atm4 = pmd_mol.atoms[atom_idx[3]]
            # Loop through all of the atoms
            ignore_end = (atm1 in atm4.bond_partners or
                            atm1 in atm4.angle_partners or
                            atm1 in atm4.dihedral_partners)
            new_dih = Dihedral(atm1, atm2, atm3, atm4, improper=False,
                            ignore_end=ignore_end, type=new_dih_typ)
            pmd_mol.dihedrals.append(
                new_dih
            )
            
    return pmd_mol