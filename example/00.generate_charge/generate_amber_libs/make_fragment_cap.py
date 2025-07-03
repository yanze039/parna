import parna
import parna.construct as construct
from pathlib import Path


na_dir = Path("../cap")
lib_dirs = list(na_dir.glob("*.lib"))

for lib in lib_dirs:
    
    name = lib.stem.split(".")[0]
    if (lib/f"{name}.lib").exists():
        print(f"Skipping {name}")
        continue
    input_mol2_file = lib/f"{name}.resp2.mol2"
    
    construct.make_cap(
        input_file=input_mol2_file,
        residue_name=name,
        output_dir=lib.resolve(),
        charge=1,
        suffix="",
    )
    