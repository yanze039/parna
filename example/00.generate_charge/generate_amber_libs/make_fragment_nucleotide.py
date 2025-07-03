import parna
import parna.construct as construct
from pathlib import Path
import os


na_dir = Path("../locked_nucleotides")
_lib_dirs = list(na_dir.glob("*.lib"))
lib_dirs = []
for x in _lib_dirs:
    mol_name = x.stem.split(".")[0]
    if not os.path.exists(x/f"{mol_name}_without_phosphate.lib"):
        lib_dirs.append(x)
print(lib_dirs)
for lib in lib_dirs:
    print(lib)
    
    name = lib.stem.split(".")[0]
    
    input_mol2_file = lib/f"{name}.resp2.mol2"
   
    if not os.path.exists(input_mol2_file):
        print(f"Skipping {name}")
        continue
   
    construct.make_fragment(
            input_file=input_mol2_file,
            residue_name=name,
            residue_type="with_phosphate",
            output_dir=lib.resolve(),
            charge=0,
            suffix="_with_phosphate",
            oxygen_type="dcase"
        )
    
    construct.make_fragment(
            input_file=input_mol2_file,
            residue_name=name,
            residue_type="without_phosphate",
            output_dir=lib.resolve(),
            charge=0,
            suffix="_without_phosphate",
            oxygen_type="dcase"
        )
    
    