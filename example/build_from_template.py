import parna
import parna.construct as construct
from parna.extract import extract_backbone
from parna.utils import read_yaml, remove_ter
from pathlib import Path


oligo_template_path = "oligo.pdb"
oligo_pdb = Path(oligo_template_path)
bk_output_dir = Path("backbone.lib")
noncanonical_residues = [1,]
sequence = ["cap", "G2", "A", "A", "A"]
output = "cap_GAAA_HB_2_2.pdb"
noter = True
attachAtoms = read_yaml(bk_output_dir/"attach_atoms.yaml")
cap_file = bk_output_dir/"CAP_1.pdb"


extract_backbone(
        str(oligo_pdb), 
        output_dir=bk_output_dir, 
        noncanonical_residues=noncanonical_residues, 
        residues_without_phosphate=[], 
        keep_hydrogen=False
    )

myS = construct.build_backbone(
    sequence, 
    baseAttachAtoms=construct.get_base_attach_atoms(),
    backboneAttachAtoms=attachAtoms,
    chain_id="A", 
    cap_file=cap_file,
    template_dir=bk_output_dir,
    local_frame_dir=bk_output_dir,
)

myS.write_pdb(output, use_hetatoms=False)

if noter:
    remove_ter(output, output)

