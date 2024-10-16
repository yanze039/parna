import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np
import parmed as pmd
import os
from parna.utils import merge_list, flatten_list, map_atoms, inverse_mapping, rd_load_file
from parna.logger import getLogger
from typing import Tuple, Union, List
from pathlib import Path
import pathlib
from pathlib import PosixPath
import yaml
import copy


def posix_path_constructor(loader, node):
    value = loader.construct_sequence(node)
    return "/".join(value)


yaml.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', posix_path_constructor)

logger = getLogger(__name__)


FUNCTIONAL_GROUPS = {
    'hydrazine': '[NX3][NX3]',
    'hydrazone': '[NX3][NX2]',
    'nitric oxide': '[N]-[O]',
    'amide': '[#7][#6](=[#8])',
    'amide-': '[#7][#6](-[O-])',
    'urea': '[NX3][CX3](=[OX1])[NX3]',
    'aldehyde': '[CX3H1](=O)[#6]',
    'sulfoxide': '[#16X3]=[OX1]',
    'sulfoxide': '[#16X3+][OX1-]',
    'sulfonyl': '[#16X4](=[OX1])(=[OX1])',
    'sulfinic acid': '[#16X3](=[OX1])[OX2H,OX1H0-]',
    'sulfinamide': '[#16X4](=[OX1])(=[OX1])([NX3R0])',
    'sulfonic acid': '[#16X4](=[OX1])(=[OX1])[OX2H,OX1H0-]',
    'phosphine oxide': '[PX4](=[OX1])([#6])([#6])([#6])',
    'phosphonate': 'P(=[OX1])([OX2H,OX1-])([OX2H,OX1-])',
    'phosphate': '[PX4](=[OX1])([#8])([#8])([#8])',
    'carboxylic acid': '[CX3](=O)[OX1H0-,OX2H1]',
    'nitro1': '[NX3](=O)(=O)',
    'nitro2': '[NX3+](=O)[O-]',
    'ester': '[CX3](=O)[OX2H0]',
    'trihalides': '[#6]([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])'
}

TERMINAL_HYDROGENS = {
    'hydroxyl': '[OX2H]',
    'amine': '[NX3H2]',
    'thiol': '[SX2H]',
    'methyl': '[CX4H3]',
    'amine+': '[NX4H3]',
}

BOND_ORDERS = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.0  # Aromatic bond is considered as single bond
}


class Fragment:
    def __init__(self,
                 mol: Chem.rdchem.Mol = None,
                 rotatable_bond: Tuple[int, int] = None,
                 dihedral_quartets: dict = None,
                 parent_fragment_mapping = None,
                 charge=None
    ):
        """Fragment class to store fragment information
        Args:
            mol (Chem.rdchem.Mol): RDKit molecule object
            rotatable_bond (Tuple[int, int]): Rotatable bond indices of the `Fragment`.
            dihedral_quartets (dict): Dihedral quartets (Index from `Fragment`)
            parent_fragment_mapping (dict): Atom mapping. Keys: Praent atom index, Values: Fragment atom index
            charge (int): Charge of the fragment
        """
        self.mol = mol
        self.rotatable_bond = rotatable_bond
        self.dihedral_quartets = dihedral_quartets
        self.parent_fragment_mapping = parent_fragment_mapping
        self.charge = charge
    
    @property
    def pivotal_dihedral_quartet(self):
        if len(self.dihedral_quartets["hetereo-hetereo"]) > 0:
            return self.dihedral_quartets["hetereo-hetereo"]
        elif len(self.dihedral_quartets["hetereo-carbon"]) > 0:
            return self.dihedral_quartets["hetereo-carbon"]
        elif len(self.dihedral_quartets["carbon-carbon"]) > 0:
            return self.dihedral_quartets["carbon-carbon"]
        else:
            return None
    
    @property
    def all_dihedral_quartets(self):
        return self.dihedral_quartets["all"]
    
    @property
    def fragment_parent_mapping(self):
        return inverse_mapping(self.parent_fragment_mapping)

    def save(self, output_dir, format="pdb"):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        if format == "sdf":
            mol_file = output_dir/"fragment.sdf"
            writer = Chem.SDWriter(str(mol_file))
            writer.write(self.mol)
            writer.close()
        elif format == "pdb":
            mol_file = output_dir/"fragment.pdb"
            tmp_mol_file = output_dir/"fragment.tmp.sdf"
            writer = Chem.SDWriter(str(tmp_mol_file))
            writer.write(self.mol)
            writer.close()
            pmd_mol = pmd.load_file(str(tmp_mol_file))[0]
            pmd_mol.residues[0].name = "MOL"
            pmd_mol.write_pdb(str(mol_file), use_hetatoms=False)
            os.remove(tmp_mol_file)
            # Chem.MolToPDBFile(self.mol, str(mol_file), flavor=10)  # do not write CONECT records
        else:
            raise ValueError("Format not supported.")
        with open(output_dir/"fragment.yaml", "w") as f:
            yaml.dump({
                "mol_file": str(mol_file.resolve()),
                "rotatable_bond": list(self.rotatable_bond),
                "dihedral_quartets": self.dihedral_quartets,
                "parent_fragment_mapping": self.parent_fragment_mapping,
                "charge": self.charge
            }, f)
    
    def load(self, filename):
        with open(filename, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.mol = rd_load_file(data["mol_file"], charge=int(data["charge"]))
        self.rotatable_bond = tuple(data["rotatable_bond"])
        self.dihedral_quartets = data["dihedral_quartets"]
        self.parent_fragment_mapping = data["parent_fragment_mapping"]
        self.charge = data["charge"]
        
        
class TorsionFragmentizer:
    def __init__(self, 
                 mol: Chem.rdchem.Mol,
                 cap_methylation : bool = True,
                 rotatable_bonds_smarts : str ="[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]",
                 non_14_hydrogen: bool = True,
                 determine_bond_orders: bool = True,
                 break_aromatic_ring: bool = True
                 ):
        self.mol = copy.deepcopy(mol)
        self.rotatable_bonds_smarts = rotatable_bonds_smarts
        self.cap_methylation = cap_methylation
        self.sssr = [list(r) for r in Chem.GetSymmSSSR(self.mol)]
        self._get_seperate_rings()
        self.functional_groups = FUNCTIONAL_GROUPS
        self.functional_mols = [Chem.MolFromSmarts(fg) for fg in self.functional_groups.values()]
        self.functional_matches = []
        for fg_mol in self.functional_mols:
            matches = self.mol.GetSubstructMatches(fg_mol)
            if len(matches) != 0:
                self.functional_matches.append(matches)
        self.terminal_hydrogens = TERMINAL_HYDROGENS
        self.terminal_mols = [Chem.MolFromSmarts(th) for th in self.terminal_hydrogens.values()]
        self.terminal_matches = []
        for th_mol in self.terminal_mols:
            matches = self.mol.GetSubstructMatches(th_mol)
            if len(matches) != 0:
                self.terminal_matches.extend(list(matches))
        self.terminal_matches = flatten_list(self.terminal_matches)
        self.non_14_hydrogen = non_14_hydrogen
        self.determine_bond_orders = determine_bond_orders
        self.break_aromatic_ring = break_aromatic_ring
        
    def get_sssr(self):
        return self.sssr
    
    def _get_seperate_rings(self):
        if len(self.sssr) == 0:
            self.rings = []
            return self.rings
        elif len(self.sssr) == 1:
            self.rings = self.sssr
            return self.rings
        else:
            self.rings = [self.sssr[0]]
            for new_ring in self.sssr[1:]:
                existance = np.array([(not set(new_ring).isdisjoint(existing_ring)) for existing_ring in self.rings])
                if np.sum(existance) == 0:
                    self.rings.append(new_ring)
                elif np.sum(existance) == 1:
                    index = np.where(existance)[0][0]
                    self.rings[index] = merge_list(self.rings[index], new_ring)
                elif np.sum(existance) > 1:
                    index = sorted(np.where(existance)[0], reverse=True)
                    for ii in index[:-1]:
                        self.rings[index[-1]] = merge_list(self.rings[index[-1]], self.rings[ii])
                        self.rings.pop(ii)
                    self.rings[index[-1]] = merge_list(self.rings[index[-1]], new_ring)
            return self.rings
    
    def get_seperate_rings(self):
        if hasattr(self, 'rings') is not None:
            return self.rings
        else:
            self._get_seperate_rings()
        
    def get_rotatable_bonds(self):
        rotatable_bond = Chem.MolFromSmarts(self.rotatable_bonds_smarts)
        _matches = self.mol.GetSubstructMatches(rotatable_bond)
        if self.non_14_hydrogen:
            matches = []
            for match in _matches:
                if any([m in self.terminal_matches for m in match]):
                    continue
                matches.append(match)
        else:
            matches = _matches
        return matches
    
    def fragmentize(self):
        self.rotatable_bonds = self.get_rotatable_bonds()
        logger.info(f"Found {len(self.rotatable_bonds)} rotatable bonds.")
        self._fragments = self.fragment_on_bonds(self.rotatable_bonds)
            
    @property
    def fragments(self):
        if hasattr(self, '_fragments'):
            return self._fragments
        else:
            self.fragmentize()
            return self._fragments
    
    # def get_fragment_rotamer_bond(self):
    #     return self.fragment_rotamer_bond
    
    # def get_mapping_dict(self):
    #     return self.mapping_dict
    
    def _get_fragment_atom_mapping(self):
        """
        Deprecated. Because `map_atoms` function is too slow for large molecules.
        We can infer the atom mapping from the fragmentation. 
        """
        logger.info("Getting fragment atom mapping.")
        self.fragment_atom_mapping = []
        for fragment_mol in self.fragment_mols:
            self.fragment_atom_mapping.append(map_atoms(self.mol, fragment_mol))
        return self.fragment_atom_mapping
    
    def get_fragment_atom_mapping(self):
        if hasattr(self, 'fragment_atom_mapping'):
            return self.fragment_atom_mapping
        else:
            self._get_fragment_atom_mapping()
            return self.fragment_atom_mapping

    def fragment_on_bonds(self, bonds):
        """
        Reference:
        [1] Capturing non-local through-bond effects in molecular mechanics force fields I: 
            Fragmenting molecules for quantum chemical torsion scans 
            https://doi.org/10.1101/2020.08.27.270934
            
        [2] Comprehensive Assessment of Torsional Strain in Crystal Structures of Small Molecules 
            and Protein-Ligand Complexes using ab Initio Calculations
            https://doi.org/10.1021/acs.jcim.9b00373
        
        Pfizer rules for fragmenting molecules:
            1. Find acyclic bond. For this step we use the SMARTS pattern [!$(*#*)!D1]-,=;!@[!$(*#*)&!D1]. But we exclude `=` here.
            2. Keep the four atoms in the torsion quartet and all atoms bonded to those atoms. This ensures that all 
               1-5 atoms are included in the minimal fragment.
            3. If any of the atoms are part of a ring or functional group shown in Table 1, include ring and functional 
               groups atoms to avoid ring breaking and fragmenting functional groups that contain more than one 
               heteroatom. The list in Table 1 are the functional groups that were present in the validation and not exhaustive.
            4. Keep all ortho substitutents relative to the torsion quartet atoms.
            5. N, O and S are capped with methyl. All other open valence atoms are capped with hydrogen
            6. We ignore the ELF10 WBO threshold for the fragmenting step mentioned in Ref[1].
        """
        fragments = []
        for bond in bonds:
            fragment = self.fragment_on_bond(bond[0], bond[1])            
            fragments.append(fragment)
        return fragments
    
    
    def fragment_on_bond(self, atomIdx1: int, atomIdx2: int, keep_tagged_atoms: bool = True):
        fragment_mol = self._fragment_on_bond(atomIdx1, atomIdx2, keep_tagged_atoms=keep_tagged_atoms)            
        parent_frgament_mapping = {}
        for i, atom in enumerate(fragment_mol.GetAtoms()):
            if atom.HasProp("_ParentIndex"):
                parent_frgament_mapping[int(atom.GetProp("_ParentIndex"))] = i
        
        fragment_rotamer_bond = (parent_frgament_mapping[atomIdx1], parent_frgament_mapping[atomIdx2])
        
        fragment = Fragment(
            fragment_mol, 
            fragment_rotamer_bond, 
            self.get_dihedral_quartet_from_bond(fragment_mol, fragment_rotamer_bond),
            parent_frgament_mapping,
            Chem.GetFormalCharge(fragment_mol)
        )
        return fragment
                
    def _fragment_on_bond(self, atomIdx1: int, atomIdx2: int, keep_tagged_atoms: bool = True):
        """
        Fragmentize molecule on a single bond. Return the RDMol object.
        """ 
        logger.info(f"Fragmentizing molecule on bond:  {atomIdx1}, {atomIdx2}")
        # rule 1 get the bond
        atom_list = [atomIdx1, atomIdx2]
        # rule 2 get 1-4 atom and 1-5 atoms
        for atom_idx in [atomIdx1, atomIdx2]:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor1 in atom.GetNeighbors():
                if neighbor1.GetIdx() not in atom_list:
                    atom_list.append(neighbor1.GetIdx())
                for neighbor2 in neighbor1.GetNeighbors():
                    if neighbor2.GetIdx() not in atom_list:
                        atom_list.append(neighbor2.GetIdx())
        logger.info(f"Atom list after adding 1-5 atoms: {atom_list}")
        
        # rule 3.1 get ring atoms. Also include heavy atoms linked to the ring.
        atom_list_15 = atom_list.copy()
        for ring in self.rings:
            if set(atom_list_15).intersection(set(ring)):
                atom_list = merge_list(atom_list, ring)
                for ratom in ring:
                    atom = self.mol.GetAtomWithIdx(ratom)
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIdx() not in atom_list:
                            atom_list.append(neighbor.GetIdx())
        logger.info(f"Atom list after adding rings: {atom_list}")
        
        # rule 3.2 get functional group atoms
        for fg_match in self.functional_matches:
            if set(atom_list).intersection(set(fg_match)):
                atom_list = merge_list(atom_list, fg_match)
        logger.info(f"Atom list after adding functional groups: {atom_list}")
        # rule 4 get ortho substitutents, we already included all atoms linked to the 
        # ring. So, we don't need to do anything here.
        
        # fragmentation
        if len(atom_list) == self.mol.GetNumAtoms():
            logger.warning("Fragments are the same as the molecule. Returning the molecule.")
            return self.mol
        
        # find the bonds to break
        if not self.break_aromatic_ring:
            atom_list = self.exclude_breaking_aromatic_bonds(self.mol, atom_list)
            logger.info(f"Atom list after excluding aromatic bonds: {atom_list}")
        atom_list = self.exclude_breaking_heteroatom_pairs(self.mol, atom_list)
        logger.info(f"Atom list after excluding heteroatom pairs: {atom_list}")
        
        if keep_tagged_atoms:
            for atom in self.mol.GetAtoms():
                if atom.HasProp("_keep_at_fragment"):
                    if atom.GetProp("_keep_at_fragment") == "1":
                        if atom.GetIdx() not in atom_list:
                            atom_list.append(atom.GetIdx())
            logger.info(f"Atom list after keeping tagged atoms: {atom_list}")
        
        atom_list.sort()
        
        breaking_bond_idx_list, breaking_bond_type_list = self.find_breaking_bonds(self.mol, atom_list)
        
        if len(breaking_bond_idx_list) == 0:
            raise RuntimeError("No bond found to break")
        logger.info(f"Found {len(breaking_bond_idx_list)} bonds to break.")        
        
        frags_dummy = Chem.FragmentOnBonds(self.mol, breaking_bond_idx_list, addDummies=True, bondTypes=breaking_bond_type_list)
        frags_assigned = []
        frags_mol_atom_mapping = []
        frags_mols = Chem.GetMolFrags(
            frags_dummy, asMols=True, sanitizeFrags=False, 
            frags=frags_assigned, fragsMolAtomMapping=frags_mol_atom_mapping)
        
        edit = Chem.RWMol(frags_mols[frags_assigned[atom_list[0]]])
        for new_index, original_index in enumerate(atom_list):
            assert original_index == frags_mol_atom_mapping[frags_assigned[atom_list[0]]][new_index], \
                f"{original_index} != {frags_mol_atom_mapping[frags_assigned[atom_list[0]]][new_index]}"
            edit.GetAtomWithIdx(new_index).SetProp("_ParentIndex", str(original_index))
        
        methyl_caps = []
        for atom in edit.GetAtoms():
            if atom.GetSymbol() == '*':
                
                dummy_neighbor = atom.GetNeighbors()[0]
                bond_type = edit.GetBondBetweenAtoms(atom.GetIdx(), dummy_neighbor.GetIdx()).GetBondType()
                # print("Cap Neighbor", dummy_neighbor.GetIdx(), dummy_neighbor.GetSymbol())
                # print("dummy Isotope", atom.GetIsotope())
                # print("dummy Neighbor Isotope", dummy_neighbor.GetIsotope())
                if self.cap_methylation and dummy_neighbor.GetAtomicNum() in [7, 8, 16]:
                    cap = Chem.Atom(6)
                    cap.SetProp("_Cap", "C")
                    cap.SetProp("_NumHs", str(int(4-BOND_ORDERS[bond_type])))
                    methyl_caps.append(atom.GetIdx())
                    edit.ReplaceAtom(atom.GetIdx(), cap, updateLabel=True, preserveProps=False)
                    if dummy_neighbor.GetAtomicNum() == 6:
                        dummy_neighbor.SetProp("_NumHs", str(1))
                else:
                    cap = Chem.Atom(1)
                    cap.SetProp("_Cap", "H")
                    if self.mol.GetAtomWithIdx(atom.GetIsotope()).GetAtomicNum() == 1:
                        cap.SetProp("_ParentIndex", str(atom.GetIsotope()))
                    if dummy_neighbor.HasProp("_NumHs"):
                        dummy_neighbor.SetProp("_NumHs", str(int(dummy_neighbor.GetProp("_NumHs"))+int(int(BOND_ORDERS[bond_type]-1))))
                    else:
                        dummy_neighbor.SetProp("_NumHs", str(int(BOND_ORDERS[bond_type]-1)))
                    edit.ReplaceAtom(atom.GetIdx(), cap, updateLabel=True, preserveProps=False)
                    edit.GetBondBetweenAtoms(dummy_neighbor.GetIdx(), atom.GetIdx()).SetBondType(Chem.BondType.SINGLE)
        if self.cap_methylation:
            logger.info(f"Added {len(methyl_caps)} methyl groups.")
        for atom in edit.GetAtoms():
            if atom.HasProp("_NumHs"):
                print("Cap", atom.GetIdx(), atom.GetProp("_NumHs"), atom.GetSymbol())
                atom.SetNumExplicitHs(int(atom.GetProp("_NumHs")))
                atom.SetNoImplicit(True)
            else:
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(True)
        
        # replace PH4 by CH3
        fragment = edit.GetMol()
        fragment.UpdatePropertyCache(strict=False)
        fragment = Chem.ReplaceSubstructs(fragment, 
                                 Chem.MolFromSmarts('[P;H4]'), 
                                 Chem.MolFromSmiles('C'),
                                 replaceAll=True)[0]
        
        for atom in fragment.GetAtoms():
            if not atom.IsInRing():
                if atom.GetIsAromatic():
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.BondType.AROMATIC:
                            bond.SetBondType(Chem.BondType.SINGLE)
                    n_bonds = len(list(atom.GetBonds()))
                    if atom.GetExplicitValence() - n_bonds >= 0:
                        atom.SetNumExplicitHs(atom.GetExplicitValence() - n_bonds)
                    atom.SetIsAromatic(False)
                    
        fragment = Chem.AddHs(fragment)
        Chem.SanitizeMol(fragment)
        
        if os.environ.get("DEBUG", False):
            from rdkit.Chem import Draw
            import copy
            imgmol = copy.deepcopy(fragment)
            AllChem.Compute2DCoords(imgmol)
            img = Draw.MolToImage(imgmol)
            img.save(f"fragment_{atomIdx1}_{atomIdx2}.png")
        
        logger.info(f"Fragment owns charge: {Chem.GetFormalCharge(fragment)}")
        if self.determine_bond_orders:
            Chem.rdDetermineBonds.DetermineBonds(fragment, charge=Chem.GetFormalCharge(fragment))
        AllChem.EmbedMolecule(fragment, randomSeed=1106)
        AllChem.MMFFOptimizeMolecule(fragment)
        
        return fragment
    
    def exclude_breaking_aromatic_bonds(self, mol: Chem.rdchem.Mol, atom_list: List[int]):
        breaking_bond_idx_list, breaking_bond_type_list = self.find_breaking_bonds(mol, atom_list)
        while Chem.rdchem.BondType.AROMATIC in breaking_bond_type_list:
            for i in range(len(breaking_bond_idx_list)):
                if breaking_bond_type_list[i] == Chem.rdchem.BondType.AROMATIC:
                    bond = mol.GetBonds()[breaking_bond_idx_list[i]]
                    for ring in self.rings:
                        if set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]).intersection(set(ring)):
                            atom_list = merge_list(atom_list, ring)
                            for ratom in ring:
                                atom = mol.GetAtomWithIdx(ratom)
                                for neighbor in atom.GetNeighbors():
                                    if neighbor.GetIdx() not in atom_list:
                                        atom_list.append(neighbor.GetIdx())
            breaking_bond_idx_list, breaking_bond_type_list = self.find_breaking_bonds(mol, atom_list)
        return atom_list
    
    def exclude_breaking_heteroatom_pairs(self, mol: Chem.rdchem.Mol, atom_list: List[int]):
        breaking_bond_idx_list, _ = self.find_breaking_bonds(mol, atom_list)
        for i in range(len(breaking_bond_idx_list)):
            bond = mol.GetBonds()[breaking_bond_idx_list[i]]
            bond_partner_idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            if 15 in [mol.GetAtomWithIdx(j).GetAtomicNum() for j in bond_partner_idx]:
                logger.info(f"Found breaking bond containing phosphorus")
                atom_in_list_index = np.where([j in atom_list for j in bond_partner_idx])[0].item()
                if mol.GetAtomWithIdx(atom_in_list_index).GetAtomicNum() == 15:
                    atom_list = merge_list(atom_list, bond_partner_idx)
                    continue
                else:
                    continue
            
            if all([mol.GetAtomWithIdx(j).GetAtomicNum() not in [5, 6, 1] for j in bond_partner_idx]):
                atom_list = merge_list(atom_list, bond_partner_idx)
        return atom_list
        
    def find_breaking_bonds(self, mol: Chem.rdchem.Mol, atom_list: List[int]):
        breaking_bond_idx_list = []
        breaking_bond_type_list = []
        for atom_idx in atom_list:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in atom_list:
                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                    breaking_bond_idx_list.append(bond.GetIdx())
                    breaking_bond_type_list.append(bond.GetBondType())
        return breaking_bond_idx_list, breaking_bond_type_list
    
    def get_dihedral_quartet_from_bond(self, mol: Chem.rdchem.Mol, bond: Union[Tuple[int, int], List[int]]):
        """
        Get the dihedral quartet from a bond
        """
        dihedral_quartet = {
            "hetereo-hetereo": [],  # heteroatom - heteroatom
            "hetereo-carbon": [],   # heteroatom - carbon
            "carbon-carbon": [],    # carbon - carbon
            "all" : []              # all atoms
        }
        atom1 = mol.GetAtomWithIdx(bond[0])
        atom2 = mol.GetAtomWithIdx(bond[1])
        neighbor1_dict = {"hetereoatom": [], "carbon": [], "hydrogen": []}
        neighbor2_dict = {"hetereoatom": [], "carbon": [], "hydrogen": []}
        for neighbor1 in atom1.GetNeighbors():
            if neighbor1.GetIdx() != atom2.GetIdx():
                if neighbor1.GetAtomicNum() == 1:
                    neighbor1_dict["hydrogen"].append(neighbor1)
                elif neighbor1.GetAtomicNum() == 6:
                    neighbor1_dict["carbon"].append(neighbor1)
                else:
                    neighbor1_dict["hetereoatom"].append(neighbor1)
        for neighbor2 in atom2.GetNeighbors():
            if neighbor2.GetIdx() != atom1.GetIdx():
                if neighbor2.GetAtomicNum() == 1:
                    neighbor2_dict["hydrogen"].append(neighbor2)
                elif neighbor2.GetAtomicNum() == 6:
                    neighbor2_dict["carbon"].append(neighbor2)
                else:
                    neighbor2_dict["hetereoatom"].append(neighbor2)
        
        for hetereoatom1 in neighbor1_dict["hetereoatom"]:
            for hetereoatom2 in neighbor2_dict["hetereoatom"]:
                dihedral_quartet["hetereo-hetereo"].append((hetereoatom1.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), hetereoatom2.GetIdx()))
        for hetereoatom in neighbor1_dict["hetereoatom"]:
            for carbon in neighbor2_dict["carbon"]:
                dihedral_quartet["hetereo-carbon"].append((hetereoatom.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), carbon.GetIdx()))
        for carbon in neighbor1_dict["carbon"]:
            for hetereoatom in neighbor2_dict["hetereoatom"]:
                dihedral_quartet["hetereo-carbon"].append((carbon.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), hetereoatom.GetIdx()))
        for carbon1 in neighbor1_dict["carbon"]:
            for carbon2 in neighbor2_dict["carbon"]:
                dihedral_quartet["carbon-carbon"].append((carbon1.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), carbon2.GetIdx()))
        
        for category1 in neighbor1_dict:
            for category2 in neighbor2_dict:
                for atom0 in neighbor1_dict[category1]:
                    for atom3 in neighbor2_dict[category2]:
                        dihedral_quartet["all"].append((atom0.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), atom3.GetIdx()))
        assert len(dihedral_quartet["all"]) > 0, "No dihedral quartet found."
        
        return dihedral_quartet

    def save_fragments(self, filename: str = "fragments.sdf"):
        writer = Chem.SDWriter(filename)
        for frag in self.fragments:
            writer.write(frag.mol)
        writer.close()
    
    def save(self, mol, filename: str = "fragments.sdf"):
        if filename.split(".")[-1] == "sdf":
            writer = Chem.SDWriter(filename)
            writer.write(mol)
            writer.close()
        elif filename.split(".")[-1] == "pdb":
            writer = Chem.PDBWriter(filename)
            writer.write(mol)
            writer.close()
        else:
            raise ValueError("File format not supported.")
        

class C5Fragmentizer:
    def __init__(self, 
                 mol: Chem.rdchem.Mol,
                 C1p_index: int,
                 O5p_index: int,
                 C5p_index: int,
                 rotatable_bonds_smarts : str ="[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]",
                 ):
        self.mol = mol
        self.C1p_index = C1p_index
        self.O5p_index = O5p_index
        self.C5p_index = C5p_index
        self.bond = self.mol.GetBondBetweenAtoms(self.O5p_index, self.C5p_index)
        self.bond_index = self.bond.GetIdx()
        self.rotatable_bonds_smarts = rotatable_bonds_smarts
        self.non_14_hydrogen = True
        self.terminal_hydrogens = TERMINAL_HYDROGENS
        self.terminal_mols = [Chem.MolFromSmarts(th) for th in self.terminal_hydrogens.values()]
        self.terminal_matches = []
        for th_mol in self.terminal_mols:
            matches = self.mol.GetSubstructMatches(th_mol)
            if len(matches) != 0:
                self.terminal_matches.extend(list(matches))
        self.terminal_matches = flatten_list(self.terminal_matches)
    
    def get_rotatable_bonds(self):
        rotatable_bond = Chem.MolFromSmarts(self.rotatable_bonds_smarts)
        _matches = self.mol.GetSubstructMatches(rotatable_bond)
        if self.non_14_hydrogen:
            matches = []
            for match in _matches:
                if any([m in self.terminal_matches for m in match]):
                    continue
                matches.append(match)
        else:
            matches = _matches
        return matches
    
    def fragment_on_bond(self):
        frags_dummy = Chem.FragmentOnBonds(self.mol, [self.bond_index], addDummies=True, bondTypes=[self.bond.GetBondType()])
        frags_assigned = []
        frags_mol_atom_mapping = []
        frags_mols = Chem.GetMolFrags(
            frags_dummy, asMols=True, sanitizeFrags=False, 
            frags=frags_assigned, fragsMolAtomMapping=frags_mol_atom_mapping)
        edit = Chem.RWMol(frags_mols[frags_assigned[self.C5p_index]])
        for atom in edit.GetAtoms():
            if atom.GetSymbol() == '*':
                cap = Chem.Atom(1)
                cap.SetProp("_Cap", "H")
                edit.ReplaceAtom(atom.GetIdx(), cap, updateLabel=True, preserveProps=False)
        for atom in edit.GetAtoms():
            # set explicit hydroms
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
        fragment = edit.GetMol()
        fragment.UpdatePropertyCache(strict=False)
        for atom in fragment.GetAtoms():
            if not atom.IsInRing():
                if atom.GetIsAromatic():
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.BondType.AROMATIC:
                            bond.SetBondType(Chem.BondType.SINGLE)
                    n_bonds = len(list(atom.GetBonds()))
                    if atom.GetExplicitValence() - n_bonds >= 0:
                        atom.SetNumExplicitHs(atom.GetExplicitValence() - n_bonds)
                    atom.SetIsAromatic(False)
        fragment = Chem.AddHs(fragment)
        Chem.SanitizeMol(fragment)
        AllChem.EmbedMolecule(fragment, randomSeed=1106)
        return fragment
    
    def fragmentize(self):
        self.rotatable_bonds = self.get_rotatable_bonds()
        self.rotatable_bond_C1p = None
        for rotatable_bond in self.rotatable_bonds:
            if self.C1p_index in rotatable_bond:
                self.rotatable_bond_C1p = rotatable_bond
                break
        self.fragment_mol = self.fragment_on_bond()
        self.fragment_atom_mapping = map_atoms(self.mol, self.fragment_mol)
        mapping_dict = {}
        for j, k in self.fragment_atom_mapping:
            mapping_dict[j] = k
        self.mapping_dict = mapping_dict
            
    @property
    def fragment(self):
        if self.rotatable_bond_C1p is None:
            raise RuntimeError("C1' rotatable bond not found.")
        rbond = (self.mapping_dict[self.rotatable_bond_C1p[0]], self.mapping_dict[self.rotatable_bond_C1p[1]])
        return Fragment(
                self.fragment_mol, 
                rbond, 
                self.get_dihedral_quartet_from_bond(self.fragment_mol, rbond),
                self.mapping_dict,
                Chem.GetFormalCharge(self.fragment_mol)
            ) 
    
    def get_dihedral_quartet_from_bond(self, mol: Chem.rdchem.Mol, bond: Union[Tuple[int, int], List[int]]):
        """
        Get the dihedral quartet from a bond
        """
        dihedral_quartet = {
            "hetereo-hetereo": [],  # heteroatom - heteroatom
            "hetereo-carbon": [],  # heteroatom - carbon
            "carbon-carbon": [],  # carbon - carbon
        }
        atom1 = mol.GetAtomWithIdx(bond[0])
        atom2 = mol.GetAtomWithIdx(bond[1])
        neighbor1_dict = {"hetereoatom": [], "carbon": [], "hydrogen": []}
        neighbor2_dict = {"hetereoatom": [], "carbon": [], "hydrogen": []}
        for neighbor1 in atom1.GetNeighbors():
            if neighbor1.GetIdx() != atom2.GetIdx():
                if neighbor1.GetAtomicNum() == 1:
                    neighbor1_dict["hydrogen"].append(neighbor1)
                elif neighbor1.GetAtomicNum() == 6:
                    neighbor1_dict["carbon"].append(neighbor1)
                else:
                    neighbor1_dict["hetereoatom"].append(neighbor1)
        for neighbor2 in atom2.GetNeighbors():
            if neighbor2.GetIdx() != atom1.GetIdx():
                if neighbor2.GetAtomicNum() == 1:
                    neighbor2_dict["hydrogen"].append(neighbor2)
                elif neighbor2.GetAtomicNum() == 6:
                    neighbor2_dict["carbon"].append(neighbor2)
                else:
                    neighbor2_dict["hetereoatom"].append(neighbor2)
        
        for hetereoatom1 in neighbor1_dict["hetereoatom"]:
            for hetereoatom2 in neighbor2_dict["hetereoatom"]:
                dihedral_quartet["hetereo-hetereo"].append((hetereoatom1.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), hetereoatom2.GetIdx()))
        for hetereoatom in neighbor1_dict["hetereoatom"]:
            for carbon in neighbor2_dict["carbon"]:
                dihedral_quartet["hetereo-carbon"].append((hetereoatom.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), carbon.GetIdx()))
        for carbon in neighbor1_dict["carbon"]:
            for hetereoatom in neighbor2_dict["hetereoatom"]:
                dihedral_quartet["hetereo-carbon"].append((carbon.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), hetereoatom.GetIdx()))
        for carbon1 in neighbor1_dict["carbon"]:
            for carbon2 in neighbor2_dict["carbon"]:
                dihedral_quartet["carbon-carbon"].append((carbon1.GetIdx(), atom1.GetIdx(), atom2.GetIdx(), carbon2.GetIdx()))
        return dihedral_quartet

