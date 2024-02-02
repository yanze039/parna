# PARNA

## Environment

### Conda

- psi4. For RESP calculation.
- rdkit. For molecule manipulation.
- parmed. For molecule parameterization.

### Third-party software

- Multiwfn. For RESP calculation.
- xtb. For semi-empirical calculation, energy minimization and dihedral angle scanning.
- AmberTools. For molecule parameterization and file format conversion.

## Usage Example

### RESP

The inputfile should be a complete nucleotide, with hydroxyl groups capping on the 5' end and 3' end. The RESP workflow will generate charges with charge constraints compatiable with AMBER OL3 force field. 

```python
import parna
# from parna.resp import RESP
import parna.resp as resp

resp.RESP(
    input_file=input_pdb,
    charge=charge_of_block,
    output_dir=resname,
    residue_name=resname,
    skip_constraint=None,
    n_conformers=6,
    memory="160 GB", 
    n_threads=48, 
    method_basis="HF/6-31G*"
)

```

### Make residue/fragments

The inputfile should contain the charge information, ideally `.mol2` file. This workflow will make the residue be a building block, i.e. cutting off the hydrogens on O3' and O5'. Adding phophate according to the requierments. 

```python

import parna
import parna.construct as construct

charge_of_block = 0
resname = "OME"
skip_constraint = 0

input_mol2_file = "./OME/OME_unopt.mol2"

construct.make_fragment(
    input_file=input_mol2_file,
    residue_name=resname,
    residue_type="with_phosphate",
    output_dir=resname,
    charge=charge_of_block
)

```

### Extract Backbone

If you would like to build RNA oligos from an existing RNA structure, you may use this workflow to extract the backbone of the RNA structure and then build modified RNA oligos from this backbone. 

```python
from parna.extract import extract_backbone

input_file = "example/oligo.HB.2.rep.c0.pdb"
output_dir = "backbone.lib"
noncanonical_residues = 1

extract_backbone(
        input_file, 
        output_dir=output_dir, 
        noncanonical_residues=noncanonical_residues, 
        residues_without_phosphate=[], 
        keep_hydrogen=False
    )

```

### Nucleotide Modification / Mutation

Modify the nucleotide in the RNA oligo. The inputfile will be flexiblly aligned to the template nucleotide. A new pdb with `ATOM` tag will be generated. 

```python

from parna.construct import build_residue_pdb_block

input_mol2_file = "example/OME.mol2"
charge_of_block = 0
resname = "OME"
template_file = "example/A2.template.pdb"

pdb_block = build_residue_pdb_block(
    input_file=input_mol2_file, 
    residue_name=resname, 
    template_residue=template_file, 
    atomtype="amber"
)
print(pdb_block)

```


