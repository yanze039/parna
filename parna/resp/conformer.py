from rdkit import Chem
from pathlib import Path
from rdkit.Chem import rdMolTransforms
import os
from parna.utils import rd_load_file, atomName_to_index, map_atoms
from parna.xtb import write_xtb_input, xtb
from parna.logger import getLogger

logger = getLogger(__name__)

## dihedral angle constraints
# O4'-C4'-C3'-C2' to -36 degree
# C4'-C3'-O3'-H3T to -60 degree
# C3'-C2'-O2'-HO2' to -120 degree
# C4'-C5'-O5'-HO5' to 180 degree
dihedral_constraint_templates = [
    ["O4'", "C4'", "C3'", "C2'", -36.0],
    ["C4'", "C3'", "O3'", "HO3'", -120.0],
    ["C3'", "C2'", "O2'", "HO2'", -120.0],
    ["C4'", "C5'", "O5'", "HO5'", 180.0]
]


def split_xyz_file(input_file_path, output_folder, output_prefix="conformer"):
    """
    Split a multi-conformer XYZ file into individual XYZ files.

    Parameters:
    - input_file_path (str): Path to the multi-conformer XYZ file.
    - output_folder (str): Folder where individual XYZ files will be saved.

    Returns:
    - None
    """
    # Read the content of the multi-conformer XYZ file
    with open(input_file_path, 'r') as multi_xyz_file:
        lines = multi_xyz_file.readlines()

    # Find the number of conformers
    block_length = int(lines[0].strip()) + 2
    num_conformers = len(lines) // block_length


    # Iterate over conformers and create individual XYZ files
    for conformer_index in range(num_conformers):
        # Extract the data for the current conformer
        start_index = (conformer_index) * block_length
        end_index = start_index + block_length
        conformer_data = lines[start_index:end_index]

        # Generate the output file path for the current conformer
        output_file_path = f"{output_folder}/{output_prefix}_{conformer_index}.xyz"

        # Write the conformer data to the individual XYZ file
        with open(output_file_path, 'w') as individual_xyz_file:
            individual_xyz_file.writelines(conformer_data)
    

def gen_conformer(query_file, scan_steps=6, charge=0, output_dir="conformers", skip_constraint=None):
    template_dir = Path(__file__).parent.parent/"template"
    template_file = template_dir/"sugar_template.pdb"
    query_file = Path(query_file)
    scan_steps = scan_steps
    charge = charge
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    # Load the template
    rd_template = Chem.MolFromPDBFile(str(template_file), removeHs=False)
    rd_query = rd_load_file(query_file, charge=charge)
    # Find the mapping
    atom_maps = map_atoms(rd_template, rd_query)
    mapping_dict = {}
    for atom in atom_maps:
        mapping_dict[atom[0]] = atom[1]
    name2idx = atomName_to_index(rd_template)
    
    # find the atom connecting C1', exclude those in the mapping
    # case 1: N9 in template get the corresponding atom in query
    # case 2: no mapping for N9 in template, find the atom connecting C1' in query
    # the scanning atoms are: O4'-C1'-base_atom_1-base_atom_2
    # case 1:
    if name2idx["N9"] in mapping_dict.keys():
        base_atom_1 = mapping_dict[name2idx["N9"]]
    # case 2:
    else:
        C1_idx_in_query = mapping_dict[name2idx["C1'"]]
        C1_neighbors = rd_query.GetAtomWithIdx(C1_idx_in_query).GetNeighbors()
        C1_neighbors_idx = [atom.GetIdx() for atom in C1_neighbors]
        base_atom_1_candidate = [idx for idx in C1_neighbors_idx if idx not in mapping_dict.values()]
        if len(base_atom_1_candidate) != 1:
            raise ValueError("The number of base atom 1 is not 1")
        base_atom_1 = base_atom_1_candidate[0]

    base_atom_1_neighbors = rd_query.GetAtomWithIdx(base_atom_1).GetNeighbors()
    base_atom_2_candidate = [atom.GetIdx() for atom in base_atom_1_neighbors if atom.GetIdx() not in mapping_dict.values()]
    if len(base_atom_2_candidate) == 0:
        raise ValueError("No valid neighbors of base atom 1 found")
    base_atom_2 = base_atom_2_candidate[0]

    dihedral_atoms = []
    dihedral_angles = []
    for idx, diha in enumerate(dihedral_constraint_templates):
        if idx == skip_constraint:
            continue
        _found = True
        _constraint = []
        for atom in diha[:4]:
            if not name2idx[atom] in mapping_dict.keys():
                _found = False
                break
            _constraint.append(mapping_dict[name2idx[atom]])
        if not _found:
            continue
        dihedral_angles.append(diha[4])
        dihedral_atoms.append(_constraint)

    scan_atoms = []
    scan_atoms.append(mapping_dict[name2idx["O4'"]])
    scan_atoms.append(mapping_dict[name2idx["C1'"]])
    scan_atoms.append(base_atom_1)
    scan_atoms.append(base_atom_2)

    # get the current dihedral angle for scanning atoms
    scan_start = rdMolTransforms.GetDihedralDeg(
        rd_query.GetConformer(), 
        scan_atoms[0],
        scan_atoms[1],
        scan_atoms[2],
        scan_atoms[3]
    )
    output_dir.mkdir(exist_ok=True)

    xtb_input = write_xtb_input(dihedral_atoms, dihedral_angles, 
                                scan_atoms, scan_type="dihedral", 
                                scan_start=scan_start, scan_end=(scan_start+360-360/scan_steps), 
                                scan_steps=scan_steps,
                                force_constant=0.05)
    logger.info("xtb_input: \n"+ xtb_input)
    
    with open(output_dir/"xtb_scan.inp", "w") as f:
        f.write(xtb_input)
    
    xtb(
        coord=query_file,
        inp=output_dir/"xtb_scan.inp",
        charge=charge,
        workdir=output_dir
    )
    logger.info("Spliting xtb output...")
    if not os.path.exists(output_dir/"xtbscan.log"):
        raise FileNotFoundError("xtbscan.log not found")
    split_xyz_file(
        output_dir/"xtbscan.log", output_folder=output_dir
    )
    logger.info("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input pdb file")
    parser.add_argument("-s", "--scan_steps", type=int, default=6, help="Number of scan steps")
    parser.add_argument("-c", "--charge", type=int, default=0, help="Charge of the molecule")
    parser.add_argument("-o", "--output_dir", type=str, default="conformers", help="Output directory")
    parser.add_argument("-sc", "--skip_constraint", type=int, default=None, help="Skip the constraint")
    args = parser.parse_args()
    gen_conformer(
        query_file=args.input,
        scan_steps=args.scan_steps,
        charge=args.charge,
        output_dir=args.output_dir,
        skip_constraint=args.skip_constraint
    )
    

    

    






