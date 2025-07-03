
```bash
python 03.build_structure_with_cap.py --sequence-components $sequence_component_file \
                             --oligo-pdb $oligo_pdb_complex \
                             --backbone $backbone_complex \
                             --output-dir $output_oligo_dir_complex \
                             --generate-frcmod  # use this flag to generate frcmod files

python 03.build_structure_with_cap.py --sequence-components $sequence_component_file \
                             --oligo-pdb $oligo_pdb_ligands \
                             --backbone $backbone_ligands \
                             --output-dir $output_oligo_dir_ligands
```