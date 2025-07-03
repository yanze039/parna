import parna
import parna.resp as resp
import parna.construct as construct
from pathlib import Path


na_dir = Path("../locked_nucleotides")
lib_dirs = list(na_dir.glob("*.lib"))

for lib in lib_dirs:
    
    name = lib.stem.split(".")[0]
    

    input_mol2_file = lib/f"{name}.resp2.mol2"
   
    construct.make_fragment(
            input_file=input_mol2_file,
            residue_name=name,
            residue_type="without_phosphate",
            output_dir=lib.resolve(),
            charge=0,
            suffix="_without_phosphate",
            oxygen_type="dcase"
        )