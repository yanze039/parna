import rdkit.Chem.rdFMCS as rdFMCS
from parna.utils import map_atoms, map_atoms_openfe
from parna.logger import getLogger



logger = getLogger(__name__)


def get_softcore_region(query, template, return_atom_name=False, on_linker=False, extensive=False):
    atom_mapping = map_atoms(query, template, ringMatchesRingOnly=True, 
                             atomCompare=rdFMCS.AtomCompare.CompareElements, 
                             matchChiralTag=False,
                             completeRingsOnly=True)
    if extensive:
        atom_mapping_openfe = map_atoms_openfe(query, template, element_change=False)
        if len(atom_mapping_openfe) < len(atom_mapping):
            atom_mapping = atom_mapping_openfe
    query_common_core = [x[0] for x in atom_mapping]
    template_common_core = [x[1] for x in atom_mapping]
    if on_linker:
        extended_atom_mapping = map_atoms(query, template, ringMatchesRingOnly=True, 
                                          atomCompare=rdFMCS.AtomCompare.CompareAny, 
                                          completeRingsOnly=True)
        for q, t in extended_atom_mapping:
            if q not in query_common_core:
                if query.GetAtomWithIdx(q).GetAtomicNum() == template.GetAtomWithIdx(t).GetAtomicNum():
                    query_common_core.append(q)
                    template_common_core.append(t)
    query_unique_atoms = [i for i in range(query.GetNumAtoms()) if i not in query_common_core]
    template_unique_atoms = [i for i in range(template.GetNumAtoms()) if i not in template_common_core]
    
    if return_atom_name:
        query_unique_atoms = [query.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip() for i in query_unique_atoms]
        template_unique_atoms = [template.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip() for i in template_unique_atoms]
    return query_unique_atoms, template_unique_atoms

