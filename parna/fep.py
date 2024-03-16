from parna.utils import (
    map_atoms,
)
from parna.logger import getLogger


logger = getLogger(__name__)


def get_softcore_region(query, template, return_atom_name=False):
    atom_mapping = map_atoms(query, template, ringMatchesRingOnly=True)
    query_common_core = [x[0] for x in atom_mapping]
    query_unique_atoms = [i for i in range(query.GetNumAtoms()) if i not in query_common_core]
    template_common_core = [x[1] for x in atom_mapping]
    template_unique_atoms = [i for i in range(template.GetNumAtoms()) if i not in template_common_core]
    if return_atom_name:
        query_unique_atoms = [query.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip() for i in query_unique_atoms]
        template_unique_atoms = [template.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip() for i in template_unique_atoms]
    return query_unique_atoms, template_unique_atoms
