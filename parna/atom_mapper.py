from rdkit import Chem
from rdkit.Chem import rdFMCS
# TypedDict & Unpack fixed in .legacy:
from rdkit.Chem import rdFMCS


class CompareAny(rdFMCS.MCSAtomCompare):

    def __call__(self, p, mol1, atom1, mol2, atom2):
        if (p.MatchChiralTag and not self.CheckAtomChirality(p, mol1, atom1, mol2, atom2)):
            return False
        if (p.MatchFormalCharge and not self.CheckAtomCharge(p, mol1, atom1, mol2, atom2)):
            return False
        if (p.RingMatchesRingOnly):
            return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
        return True


class CompareAnyHeavyAtom(CompareAny):

    def __call__(self, p, mol1, atom1, mol2, atom2):
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        # Any atom, including H, matches another atom of the same type,  according to
        # the other flags
        if (a1.GetAtomicNum() == a2.GetAtomicNum()
            or (a1.GetAtomicNum() > 1 and a2.GetAtomicNum() > 1)):
            return CompareAny.__call__(self, p, mol1, atom1, mol2, atom2)
        return False


class CompareElements(rdFMCS.MCSAtomCompare):
    
    def __call__(self, p, mol1, atom1, mol2, atom2):
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        if (a1.GetAtomicNum() != a2.GetAtomicNum()):
            return False
        if (p.MatchValences and a1.GetTotalValence() != a2.GetTotalValence()):
            return False
        if (p.MatchChiralTag and not self.CheckAtomChirality(p, mol1, atom1, mol2, atom2)):
            return False
        if (p.MatchFormalCharge and not self.CheckAtomCharge(p, mol1, atom1, mol2, atom2)):
            return False
        if p.RingMatchesRingOnly:
            return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
        return True


class FuzzyElementCompareAtoms(rdFMCS.MCSAtomCompare):
   
    def __init__(
            self,
            comparison,
            custom_map,
            n_atoms_mol1,
            n_atoms_mol2
        ):
        super().__init__()  # what is p_object?
        
        if comparison == rdFMCS.AtomCompare.CompareElements:
            self.comparison = CompareElements()
        elif comparison == rdFMCS.AtomCompare.CompareAnyHeavyAtom:
            self.comparison = CompareAnyHeavyAtom()
        elif comparison == rdFMCS.AtomCompare.CompareAny:
            self.comparison = CompareAny()
        else:
            raise NotImplementedError(f"Unsupported comparison {comparison}")
        
        self.register_map(custom_map, n_atoms_mol1, n_atoms_mol2)
    
    def register_map(self, custom_map, n_atoms_mol1, n_atoms_mol2):
        """
            custom_map: [(1, 2), (2, 3), ...]
        """
        if isinstance(custom_map, list):
            self.custom_map = {x[0]: x[1] for x in custom_map}
        else:
            self.custom_map = custom_map
        if n_atoms_mol1 > n_atoms_mol2:
            self.custom_map = {y:x for x,y in self.custom_map.items()}
        

    def __call__(self,
                 parameters: rdFMCS.MCSAtomCompareParameters,
                 mol1: Chem.Mol,
                 mol1_atom_idx: int,
                 mol2: Chem.Mol,
                 mol2_atom_idx: int,
                 ) -> bool:
        if mol1_atom_idx not in self.custom_map:
            return self.comparison(parameters, mol1, mol1_atom_idx, mol2, mol2_atom_idx)
        mapped_idx: int = self.custom_map[mol1_atom_idx]
        if mapped_idx == mol2_atom_idx:
            return True
        else:
            return False


class CompareChiralElements(rdFMCS.MCSAtomCompare):
    def __init__(
            self,
            mol1_chiral_list,
            mol2_chiral_list,
        ):
        super().__init__()  # what is p_object?
        self.mol1_chiral_dict = {x[0]:x[1] for x in mol1_chiral_list}
        self.mol2_chiral_list = {x[0]:x[1] for x in mol2_chiral_list}

    def __call__(self, p, mol1, atom1, mol2, atom2):
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        if (a1.GetAtomicNum() != a2.GetAtomicNum()):
            return False
        if (p.MatchValences and a1.GetTotalValence() != a2.GetTotalValence()):
            return False
        if (not self._CheckAtomChirality(p, mol1, atom1, mol2, atom2)):
            return False
        if (p.MatchFormalCharge and not self.CheckAtomCharge(p, mol1, atom1, mol2, atom2)):
            return False
        if p.RingMatchesRingOnly:
            return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
        return True
    
    def _CheckAtomChirality(self, p, mol1, atom1, mol2, atom2):
        # Check if the atoms are chiral
        if atom1 not in self.mol1_chiral_dict or atom2 not in self.mol2_chiral_list:
            return True
        elif self.mol1_chiral_dict[atom1] != self.mol2_chiral_list[atom2]:
            return False
        else:
            return True

