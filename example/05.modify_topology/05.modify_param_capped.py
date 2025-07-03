import os
from pathlib import Path
import argparse
import parmed
from parna.utils import map_atoms, rd_load_file, atomName_to_index, map_atoms_openfe
from parna.molops import modify_torsion_parameters
import rdkit.Chem as Chem
import yaml
import numpy as np
from parmed.topologyobjects import DihedralType, Dihedral, DihedralTypeList
from parmed.topologyobjects import CmapType, Cmap
from scipy.interpolate import griddata
import scipy.ndimage
import scipy

# Define custom constructor for tuples
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

# Define custom representer for tuples
def tuple_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:python/tuple', data)

# Add the constructors to PyYAML
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
yaml.add_representer(tuple, tuple_representer)

cmap_head_template_parm_file = "/home/gridsan/ywang3/Project/Capping/Benchmarking2/binding_pose/double_ome/top/complex.mod.parm7"
cmap_tmpl_mol = parmed.load_file(cmap_head_template_parm_file)

def smooth_grid(grid, sigma=1.0):
    return scipy.ndimage.gaussian_filter(grid, sigma=sigma)  

def interpolate_2d_griddata(X, Y, Z, resolution):
    X = (X + np.pi) % (2 * np.pi) - np.pi
    Y = (Y + np.pi) % (2 * np.pi) - np.pi

    X_next_period = X + np.pi*2
    Y_next_period = Y + np.pi*2
    X_last_period = X - np.pi*2
    Y_last_period = Y - np.pi*2
    
    
    all_X = np.concatenate([X, X, X, 
                            X_next_period, X_next_period, X_next_period,
                            X_last_period, X_last_period, X_last_period]).flatten()
    all_Y = np.concatenate([Y, Y_next_period, Y_last_period,
                            Y, Y_next_period, Y_last_period,
                            Y, Y_next_period, Y_last_period,]).flatten()
    all_Z = np.concatenate([Z, Z, Z, Z, Z, Z, Z, Z, Z, ]).flatten()

    minv = -np.pi 
    maxv = +np.pi 

    mask = (all_X > minv-1.5) & (all_X < maxv+1.5) & (all_Y > minv-1.5) & (all_Y < maxv+1.5)
    all_X = all_X[mask]
    all_Y = all_Y[mask]
    all_Z = all_Z[mask]

    grid_x, grid_y = np.mgrid[minv:maxv:complex(0,resolution+1), minv:maxv:complex(0,resolution+1)] 
    grid_x = grid_x[:-1, :-1]
    grid_y = grid_y[:-1, :-1] 
    
    grid_z = griddata((all_X, all_Y), all_Z, (grid_x, grid_y), method='linear')
        
    return grid_z
    

def calculate_cmap_grid(
    cmap_file, resolution, dihedral_name_1, dihedral_name_2,
    qm_energy_factor=1.0, mm_energy_factor=1.0, smooth=None
    ):
    with open(cmap_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # fit the energy function by ggrid data
    epsilons = []
    zetas = []
    qm_energys = []
    mm_energys = []

    for conf in data.keys():
        epsilons.append(data[conf]["dihedral"][dihedral_name_1])
        zetas.append(data[conf]["dihedral"][dihedral_name_2])
        qm_energys.append(data[conf]["qm_energy"])
        mm_energys.append(data[conf]["mm_energy"])

    epsilons = np.array(epsilons)

    zetas = np.array(zetas)
    qm_energys = np.array(qm_energys) * qm_energy_factor
    mm_energys = np.array(mm_energys) * mm_energy_factor

    qm_energys = qm_energys - qm_energys.min()
    mm_energys = mm_energys - mm_energys.min()
    energy_diff = qm_energys - mm_energys

    X = epsilons 
    Y = zetas 
    Z = energy_diff
    
    resolution = 24
    grid_z = interpolate_2d_griddata(X, Y, Z, resolution)

    assert grid_z.shape == (resolution, resolution), "grid_z shape is not correct"
    
    if smooth is not None:
        grid_z = smooth_grid(grid_z, sigma=smooth)
    return grid_z


def add_dihedral(pmd_mol, k, phase, periodicity, atom_lists):
    new_dih_typ = DihedralType(
        phi_k=k ,
        per=periodicity,
        phase=phase / np.pi * 180.0,
        scee=1.200,
        scnb=2.000
    )
    exists = False
    # Do not add a duplicate dihedral type
    for dih_typ in pmd_mol.dihedral_types:
        if new_dih_typ == dih_typ:
            new_dih_typ = dih_typ
            exists = True
            break
    if not exists:
        pmd_mol.dihedral_types.append(new_dih_typ)
        new_dih_typ.list = pmd_mol.dihedral_types
    
    for atom_list in atom_lists:
        atm1 = pmd_mol.atoms[atom_list[0]]
        atm2 = pmd_mol.atoms[atom_list[1]]
        atm3 = pmd_mol.atoms[atom_list[2]]
        atm4 = pmd_mol.atoms[atom_list[3]]
        # Loop through all of the atoms
        ignore_end = (atm1 in atm4.bond_partners or
                        atm1 in atm4.angle_partners or
                        atm1 in atm4.dihedral_partners)
        new_dih = Dihedral(atm1, atm2, atm3, atm4, improper=False,
                        ignore_end=ignore_end, type=new_dih_typ)
        pmd_mol.dihedrals.append(
            new_dih
        )
        # print("add", new_dih)
    return pmd_mol


def modify_parameter_by_atom_index(
        parameter_set, pmd_mol, reisude_idx, residue_rd_mol,
        template_residue_atom_map_dict, 
    ):

    all_dihedrals = np.array(parameter_set["dihedral"])
    all_ks = np.array(parameter_set["k"])
    all_phases = np.array(parameter_set["phase"])
    all_periodicity = np.array(parameter_set["periodicity"])
    n_parm = len(all_dihedrals)

    for parm_idx in range(n_parm):
        dihedral = all_dihedrals[parm_idx]
        ks = all_ks[parm_idx]
        phases = all_phases[parm_idx]
        periodicities = all_periodicity[parm_idx]
        
        atom_idx = [template_residue_atom_map_dict[x] for x in dihedral]
        print("find dihedral", dihedral, atom_idx)
        # get atom from atom idx by rdkit
        atom_names = ([residue_rd_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in atom_idx])
        sorted_atom_names = sorted(atom_names.copy())
        dihedral_list = []
        all_atom_list = []
        for idx, dihedral in enumerate(pmd_mol.dihedrals):
            if not dihedral.atom1.residue.idx == reisude_idx or not dihedral.atom4.residue.idx == reisude_idx:
                continue
            _atom_idx = [dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]
            _atom_names = sorted([pmd_mol.atoms[x].name for x in _atom_idx])
            
            if np.all([sorted_atom_names[ii] == _atom_names[ii] for ii in range(4)]):
                all_atom_list.append(tuple(_atom_idx))
                dihedral_list.append(idx)        
        dihedral_list.sort(reverse=True)
        
        assert len(dihedral_list) > 0, f"no dihedral found for {resname} {dihedral}"
        for idx in dihedral_list:
            pmd_mol.dihedrals[idx].delete()
            del pmd_mol.dihedrals[idx]
        all_atom_list = list(set(all_atom_list))
        assert len(all_atom_list) == 1, f"multiple dihedral found for {resname} {dihedral}"
        
        for pi in range(len(ks)):    
            pmd_mol = add_dihedral(pmd_mol, ks[pi], phases[pi], periodicities[pi], all_atom_list)    
    return pmd_mol


def modify_sugar_parameters(
        pmd_mol,
        sugar_pucker_parameter_set,
        sugar_v1_atom_idx,
        sugar_v3_atom_idx,
        reisude_idx,
        residue_rd_mol
    ):
    dihedral_names = ["sugar_v1", "sugar_v3"]
    dihedral_idx = {
        "sugar_v1": sugar_v1_atom_idx,
        "sugar_v3": sugar_v3_atom_idx
    }

    all_ks = np.array(sugar_pucker_parameter_set["k"])
    all_phases = np.array(sugar_pucker_parameter_set["phase"])
    all_periodicity = np.array(sugar_pucker_parameter_set["periodicity"])
    print("modify sugar pucker")
    for j, dihedral_name in enumerate(dihedral_names):
        ks = all_ks[j]
        phases = all_phases[j]
        periodicities = all_periodicity[j]
        # get atom from atom idx by rdkit
        atom_names = ([residue_rd_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in dihedral_idx[dihedral_name]] )
        sorted_atom_names = sorted(atom_names.copy())
        
        all_atom_list = []
        dihedral_list = []
        for idx, dihedral in enumerate(pmd_mol.dihedrals):
            if (dihedral.atom1.residue.idx == reisude_idx and dihedral.atom4.residue.idx == reisude_idx): 
                _atom_idx = [dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]
                _atom_names = sorted([pmd_mol.atoms[x].name for x in _atom_idx])
                if np.all([sorted_atom_names[ii] == _atom_names[ii] for ii in range(4)]):
                    all_atom_list.append(tuple(_atom_idx))
                    dihedral_list.append(idx)

        assert len(all_atom_list) == 1, f"multiple dihedral found for {resname} sugar pucker {j}"
        dihedral_list.sort(reverse=True)
        assert len(dihedral_list) > 0, f"no dihedral found for {resname} {dihedral}"
        
        for idx in dihedral_list:
            pmd_mol.dihedrals[idx].delete()
            del pmd_mol.dihedrals[idx]
        all_atom_list = list(set(all_atom_list))
                    
        for pi in range(len(ks)):      
            pmd_mol = add_dihedral(pmd_mol, ks[pi], phases[pi], periodicities[pi], all_atom_list)
    return pmd_mol

def find_dihedral_connectivity_by_atom_names(pmd_mol, atom_names, residue_idx):
    all_atom_list = []
    for idx, dihedral in enumerate(pmd_mol.dihedrals):
        if (dihedral.atom1.residue.idx == residue_idx and dihedral.atom4.residue.idx == residue_idx):
            _atom_idx = [dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]
            _atom_names = sorted([pmd_mol.atoms[x].name for x in _atom_idx])
            # print(_atom_names)
            if np.all([sorted(atom_names)[ii] == _atom_names[ii] for ii in range(4)]):
                all_atom_list.append(tuple(_atom_idx))
                
    if len(all_atom_list) == 0:
        return None
    else:
        return all_atom_list[0]
          


def patch_cmap_parameters(
        pmd_mol, cmap_grid, resolition, residue_idx, residue_rd_mol,
        first_dihedral_angle_atom_names, second_dihedral_angle_atom_names,
        connection_shifts=[0,1]
    ):
    new_cmap_typ = CmapType(
        resolution=resolition,
        grid=cmap_grid.flatten(),
    )

    pmd_mol.cmap_types.append(new_cmap_typ)
    new_cmap_typ.list = pmd_mol.cmap_types
    
    dihedral_atom_names = {
        1: first_dihedral_angle_atom_names,
        2: second_dihedral_angle_atom_names
    }
    
    all_atom_list_1 = []
    all_atom_list_2 = []
    print(f"patch cmap for residue {residue_idx} with shift {connection_shifts}")
    for idx, dihedral in enumerate(pmd_mol.dihedrals):
        # print(dihedral.atom1.residue, dihedral.atom4.residue)
        for si in range(2):
            if (dihedral.atom1.residue.idx == residue_idx and dihedral.atom4.residue.idx == residue_idx+connection_shifts[si]) or \
                (dihedral.atom1.residue.idx == residue_idx+connection_shifts[si] and dihedral.atom4.residue.idx == residue_idx):
                _atom_idx = [dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx]
                _atom_names = sorted([pmd_mol.atoms[x].name for x in _atom_idx])
                if si == 0 and np.all([sorted(dihedral_atom_names[1])[ii] == _atom_names[ii] for ii in range(4)]):
                    all_atom_list_1.append(tuple(_atom_idx))
                if si == 1 and  np.all([sorted(dihedral_atom_names[2])[ii] == _atom_names[ii] for ii in range(4)]):
                    all_atom_list_2.append(tuple(_atom_idx))
    assert len(all_atom_list_1) > 0, f"no dihedral found for {resname} {dihedral_atom_names[1]}"
    assert len(all_atom_list_2) > 0, f"no dihedral found for {resname} {dihedral_atom_names[2]}"
            
    all_atom_list_1 = all_atom_list_1[0]
    all_atom_list_2 = all_atom_list_2[0]
    
    if all_atom_list_1[1] == all_atom_list_2[0]:
        pass
    elif all_atom_list_1[::-1][1] == all_atom_list_2[0]:
        assert all_atom_list_1[::-1][2] == all_atom_list_2[1], f"list1 {all_atom_list_1} list2 {all_atom_list_2}"
        all_atom_list_1 = all_atom_list_1[::-1]
    elif all_atom_list_1[1] == all_atom_list_2[::-1][0]:
        assert all_atom_list_1[2] == all_atom_list_2[::-1][1], f"list1 {all_atom_list_1} list2 {all_atom_list_2}"
        all_atom_list_2 = all_atom_list_2[::-1]
    elif all_atom_list_1[::-1][1] == all_atom_list_2[::-1][0]:
        assert all_atom_list_1[::-1][2] == all_atom_list_2[::-1][1], f"list1 {all_atom_list_1} list2 {all_atom_list_2}"
        all_atom_list_1 = all_atom_list_1[::-1]
        all_atom_list_2 = all_atom_list_2[::-1]
    else:
        raise ValueError(f"can't understand the atom order of list1 {all_atom_list_1} list2 {all_atom_list_2}")
    atm1 = pmd_mol.atoms[all_atom_list_1[0]]
    atm2 = pmd_mol.atoms[all_atom_list_1[1]]
    atm3 = pmd_mol.atoms[all_atom_list_1[2]]
    atm4 = pmd_mol.atoms[all_atom_list_1[3]]
    atm5 = pmd_mol.atoms[all_atom_list_2[3]]

    new_cmap = Cmap(atm1, atm2, atm3, atm4, atm5,  type=new_cmap_typ)
    pmd_mol.cmaps.append(new_cmap)

    return pmd_mol

same_parameters = {
    "GR3": "GR2",
    "AR3": "AR2",
    "CR3": "CR2",
    "UR3": "UR2",
    "ARX": "AR2",
    "CRX": "CR2",
    "URX": "UR2",
    "GRX": "GR2",
    "AR5": "AR2",
    "CR5": "CR2",
    "UR5": "UR2",
    "GR5": "GR2",
    "LN2": "LNA",
    "OA3": "OME"
}

seq = [
        "GR5", "GRX", "CRX", "ARX", "CRX", "URX", "URX", "CRX", 
        "GRX", "GRX", "URX", "GRX", "CRX", "CR3"
        ]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--parm7", type=str, help="input parm7 file")
parser.add_argument("--nresidues", type=int, help="number of residues", default=4)
parser.add_argument("--alchemical", action="store_true", help="alchemical", default=False)
# parser.add_argument("--backbone", type=str, help="input backbone dir")
parser.add_argument("--output", type=str, help="output parm7 file")
parser.add_argument("--parmpatch", type=str, help="parm patch yaml file")
args = parser.parse_args()
parm7 = Path(args.parm7)
pmd_mol = parmed.load_file(str(parm7))

parm_patch_yaml_file = Path(args.parmpatch)
output_file = Path(args.output)
# backbone_dir = Path(args.backbone)

with open(parm_patch_yaml_file, "r") as f:
    patch_set = yaml.load(f, Loader=yaml.FullLoader)

data_dir_cap = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data")
data_dir_na  = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/data")

sugar_template_file = "/home/gridsan/ywang3/Project/Capping/parna/parna/template/sugar_template.pdb"
sugar_tmpl_mol = rd_load_file(sugar_template_file, removeHs=False)
name2idx_sugar_tmpl = atomName_to_index(sugar_tmpl_mol)

solution = ["WAT", "Na+", "Cl-"]
n_residues = args.nresidues
n_residues_alchem = n_residues * 2 if args.alchemical else n_residues

lib_dir = Path("/home/gridsan/ywang3/Project/Capping/Benchmarking2/lib")

for i in range(n_residues_alchem):
    
    resname = pmd_mol.residues[i].name
    # residue_pdb = backbone_dir / f"{resname}.{(i % n_residues)+1}.pdb"
    # residue_mol = Chem.MolFromPDBFile(str(residue_pdb), removeHs=False)
   
    
    if i % n_residues == 0:
        category = "cap"
        residue_parm_sdf = data_dir_cap / category / f"{resname}.sdf"
        residue_mol2_file = lib_dir / category / f"{resname}.lib" / f"{resname}.mol2"
    elif i % n_residues == 1:
        category = "linker"
        residue_parm_sdf = "/home/gridsan/ywang3/Project/Capping/test_parameterization/test_patch/triphosphate_linker/DMTP.sdf"
        residue_mol2_file = lib_dir / category / f"{resname}.lib" / f"{resname}.mol2"
    else:
        category = "nucleotides"
        residue_parm_sdf = data_dir_na / category / f"{resname}.sdf"
        if not os.path.exists(residue_parm_sdf):
            residue_parm_sdf = data_dir_na / category / f"{same_parameters[resname]}.sdf"
        if i % n_residues == 2:
            residue_mol2_file = lib_dir / category / f"{resname}.lib" / f"{resname}_without_phosphate.mol2"
        else:
            residue_mol2_file = lib_dir / category / f"{resname}.lib" / f"{resname}_with_phosphate.mol2"
    print("residue mol2 file", residue_mol2_file)
    residue_mol = rd_load_file(residue_mol2_file, removeHs=False, atomtype="amber")
    
    # atom_maps_with_sugar = map_atoms_openfe(sugar_tmpl_mol, residue_mol, element_change=True)
    atom_maps_with_sugar = map_atoms(sugar_tmpl_mol, residue_mol, fuzzy_matching=True)
    atom_map_dict_with_sugar = {x[0]:x[1] for x in atom_maps_with_sugar}
    
    parm_template_mol = Chem.SDMolSupplier(str(residue_parm_sdf), removeHs=False)[0]
    atom_maps = map_atoms(parm_template_mol, residue_mol)
    # atom_maps = map_atoms_openf(parm_template_mol, residue_mol, element_change=True)
    atom_map_dict = {x[0]:x[1] for x in atom_maps}
    
    if resname in patch_set[category]:
        parm_info = patch_set[category][resname]
    else:
        parm_info = patch_set[category][same_parameters[resname]]
    
    if "chi" in parm_info:
        print(f"modify chi for {resname} at {i}")
        if "chi_parm_template" in parm_info:
            chi_parm_template_sdf = Path(parm_info["chi_parm_template"])
        else:
            print("using default chi parm template for ", resname, residue_parm_sdf)
            chi_parm_template_sdf = residue_parm_sdf
        chi_parm_template_mol = Chem.SDMolSupplier(str(chi_parm_template_sdf), removeHs=False)[0]
        chi_atom_maps = map_atoms(chi_parm_template_mol, residue_mol)
        maps1 = map_atoms_openfe(parm_template_mol, residue_mol, element_change=True)
        maps2 = map_atoms_openfe(chi_parm_template_mol, residue_mol, element_change=True)
        print("openfep1:", maps1)
        print("openfep2:", maps2)
        chi_atom_map_dict = {x[0]:x[1] for x in chi_atom_maps}
        
        chi_parameter_yaml = parm_info["chi"]
        with open(chi_parameter_yaml, "r") as f:
            chi_parameter_set = yaml.load(f, Loader=yaml.FullLoader)

        pmd_mol = modify_parameter_by_atom_index(
                chi_parameter_set, pmd_mol, i, residue_mol, chi_atom_map_dict
            )
    
    if "noncanonical" in parm_info:
        print(f"modify noncanonical for {resname} at {i}")
        noncanonical_parameter_yaml = parm_info["noncanonical"]
        with open(noncanonical_parameter_yaml, "r") as f:
            noncanonical_parameter_set = yaml.load(f, Loader=yaml.FullLoader)
        # noncanonical_parameter_set has a higher level of hierarchy
        for kname in noncanonical_parameter_set.keys():
            pmd_mol = modify_parameter_by_atom_index(
                    noncanonical_parameter_set[kname], pmd_mol, i, residue_mol, atom_map_dict
                )
    
    # sugar angle
    if "sugar" in parm_info:
        print(f"modify sugar for {resname} at {i}")
        sugar_parameter_yaml = parm_info["sugar"]
        with open(sugar_parameter_yaml, "r") as f:
            sugar_parameter_set = yaml.load(f, Loader=yaml.FullLoader)
        
        sugar_v1_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["O4'", "C1'", "C2'", "C3'"]]
        sugar_v3_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["O4'", "C4'", "C3'", "C2'"]]
        pmd_mol = modify_sugar_parameters(
            pmd_mol,
            sugar_parameter_set, 
            sugar_v1_atom_idx=sugar_v1_atom_idx, sugar_v3_atom_idx=sugar_v3_atom_idx,
            reisude_idx=i,
            residue_rd_mol=residue_mol
        )
    
    if "sugar_epsilon" in parm_info:
        print(f"modify sugar epsilon for {resname} at {i}")
        sugar_epsilon_cmap_parameter_file = Path(parm_info["sugar_epsilon"])
        if sugar_epsilon_cmap_parameter_file.suffix == ".npy":
            cmap_grid = np.load(sugar_epsilon_cmap_parameter_file)
        elif sugar_epsilon_cmap_parameter_file.suffix == ".yaml":
            cmap_grid = calculate_cmap_grid(
                sugar_epsilon_cmap_parameter_file, 24, "sugar", "epsilon"
            )
        else:
            raise ValueError("sugar_epsilon_cmap_parameter_file must be either .npy or .yaml")
        
        sugar_occo_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["O2'", "C2'", "C3'", "O3'"]]
        sugar_occo_atom_names = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in sugar_occo_atom_idx]
        epsilon_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["C2'", "C3'", "O3'"]]
        epsilon_atom_names = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in epsilon_atom_idx]
        epsilon_atom_names = epsilon_atom_names + ["P"]
        pmd_mol = patch_cmap_parameters(
            pmd_mol, cmap_grid, 24, i, residue_mol,
            sugar_occo_atom_names, epsilon_atom_names, 
            connection_shifts=[0,1]
        )
    
    if "epsilon_zeta" in parm_info:
        print(f"modify sugar epsilon zeta for {resname} at {i}")
        epsilon_zeta_cmap_parameter_file = Path(parm_info["epsilon_zeta"])
        if epsilon_zeta_cmap_parameter_file.suffix == ".npy":
            cmap_grid = np.load(epsilon_zeta_cmap_parameter_file)
        elif epsilon_zeta_cmap_parameter_file.suffix == ".yaml":
            smooth = None
            cmap_grid = calculate_cmap_grid(
                epsilon_zeta_cmap_parameter_file, 24, "epsilon", "zeta", smooth=smooth
            )
        else:
            raise ValueError("epsilon_zeta_cmap_parameter_file must be either .npy or .yaml")
        if "sugar_epsilon" in parm_info:
            epsilon_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["C2'", "C3'", "O3'"]]
        else:
            epsilon_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["C4'", "C3'", "O3'"]]
        epsilon_atom_names = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in epsilon_atom_idx]
        epsilon_atom_names = epsilon_atom_names + ["P"]
        zeta_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["C3'", "O3'"]]
        zeta_atom_names = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in zeta_atom_idx]
        zeta_atom_names = zeta_atom_names + ["P", "O5'"]
        pmd_mol = patch_cmap_parameters(
            pmd_mol, cmap_grid, 24, i, residue_mol,
            epsilon_atom_names, zeta_atom_names, connection_shifts=[1,1]
        )
    
    if "zeta_alpha" in parm_info:
        print(f"modify sugar zeta alpha for {resname} at {i}")
        epsilon_zeta_cmap_parameter_file = Path(parm_info["zeta_alpha"])
        if epsilon_zeta_cmap_parameter_file.suffix == ".npy":
            cmap_grid = np.load(epsilon_zeta_cmap_parameter_file)
        elif epsilon_zeta_cmap_parameter_file.suffix == ".yaml":
            cmap_grid = calculate_cmap_grid(
                epsilon_zeta_cmap_parameter_file, 24, "epsilon", "zeta"
            )
        else:
            raise ValueError("epsilon_zeta_cmap_parameter_file must be either .npy or .yaml")
        zeta_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["C3'", "O3'"]]
        zeta_atom_names = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in zeta_atom_idx]
        zeta_atom_names = zeta_atom_names + ["P", "O5'"]
        alpha_atom_idx = [atom_map_dict_with_sugar[name2idx_sugar_tmpl[x]] for x in ["O3'"]]
        _alpha_atom_names = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in alpha_atom_idx]
        
        try:
            alpha_atom_names = _alpha_atom_names + ["P", "O5'", "C5'"]
            pmd_mol = patch_cmap_parameters(
                pmd_mol, cmap_grid, 24, i, residue_mol,
                zeta_atom_names, alpha_atom_names, connection_shifts=[1,1]
            )
        except Exception as e:
            try:
                alpha_atom_names = _alpha_atom_names + ["P", "O5'", "C01"]
                pmd_mol = patch_cmap_parameters(
                    pmd_mol, cmap_grid, 24, i, residue_mol,
                    zeta_atom_names, alpha_atom_names, connection_shifts=[1,1]
                )
            except Exception as e:
                alpha_atom_names = _alpha_atom_names + ["P", "O5'", "C02"]
                pmd_mol = patch_cmap_parameters(
                    pmd_mol, cmap_grid, 24, i, residue_mol,
                    zeta_atom_names, alpha_atom_names, connection_shifts=[1,1]
                )
        # exit(0)
    
    if "symm" in parm_info:
        print(f"modify symm for {resname} at {i}")
        symm_cmap_parameter_file = Path(parm_info["symm"])
        if symm_cmap_parameter_file.suffix == ".npy":
            cmap_grid = np.load(symm_cmap_parameter_file)
        elif symm_cmap_parameter_file.suffix == ".yaml":
            cmap_grid = calculate_cmap_grid(
                symm_cmap_parameter_file, 24, "alpha", "beta"
            )
        else:
            raise ValueError("symm_cmap_parameter_file must be either .npy or .yaml")
        backbone_smarts = "[*]P[*]P[*]P[*]"
        # find substructure of the smarts
        backbone_substructure = Chem.MolFromSmarts(backbone_smarts)
        # find the atom index of the substructure
        backbone_atom_idx = residue_mol.GetSubstructMatches(backbone_substructure)[0]

        symm_bond_idx_1 = [backbone_atom_idx[1], backbone_atom_idx[2], backbone_atom_idx[3], backbone_atom_idx[4]]
        symm_bond_names_1 = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in symm_bond_idx_1]
        symm_bond_idx_2 = [backbone_atom_idx[5], backbone_atom_idx[4], backbone_atom_idx[3], backbone_atom_idx[2]]
        symm_bond_names_2 = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in symm_bond_idx_2]
        linker_template = parmed.modeller.residue.ResidueTemplate.from_residue(pmd_mol.residues[i])
        head = linker_template.head
        tail = linker_template.tail
        with_head = find_dihedral_connectivity_by_atom_names(
            pmd_mol, [head.name, *symm_bond_names_1][:4], i
        )

        if with_head is not None:
            print("with head")
            symm_bond_names_1 = [head.name] + symm_bond_names_1
            symm_bond_names_2 = [tail.name] + symm_bond_names_2
        else:
            print("without tail")
            symm_bond_names_1 = [tail.name] + symm_bond_names_1
            symm_bond_names_2 = [head.name] + symm_bond_names_2
        
        pmd_mol = patch_cmap_parameters(
            pmd_mol, cmap_grid, 24, i, residue_mol,
            symm_bond_names_1[:4], symm_bond_names_1[1:5],
            connection_shifts=[0,0]
        )
        pmd_mol = patch_cmap_parameters(
            pmd_mol, cmap_grid, 24, i, residue_mol,
            symm_bond_names_2[:4], symm_bond_names_2[1:5],
            connection_shifts=[0,0]
        )
        
    if "asymm" in parm_info:
        print(f"modify asymm for {resname} at {i}")
        asymm_cmap_parameter_file = Path(parm_info["asymm"])
        if asymm_cmap_parameter_file.suffix == ".npy":
            cmap_grid = np.load(asymm_cmap_parameter_file)
        elif asymm_cmap_parameter_file.suffix == ".yaml":
            cmap_grid = calculate_cmap_grid(
                asymm_cmap_parameter_file, 24, "alpha", "beta"
            )
        else:
            raise ValueError("asymm_cmap_parameter_file must be either .npy or .yaml")
        asymm_bond_idx = [backbone_atom_idx[1], backbone_atom_idx[2], backbone_atom_idx[3], backbone_atom_idx[4], backbone_atom_idx[5]]
        asymm_bond_names = [residue_mol.GetAtomWithIdx(x).GetPDBResidueInfo().GetName().strip() for x in asymm_bond_idx]
        pmd_mol = patch_cmap_parameters(
            pmd_mol, cmap_grid, 24, i, residue_mol,
            asymm_bond_names[:4], asymm_bond_names[1:5],connection_shifts=[0,0]
        )

# exit(0)

for item in ["CMAP_COUNT", "CMAP_RESOLUTION", "CMAP_INDEX"]:
    # amber_item = "_".join(item.split("_")[1:])
    amber_item = item
    pmd_mol.flag_list.append(amber_item)
    pmd_mol.parm_comments[amber_item] = cmap_tmpl_mol.parm_comments[item]
    pmd_mol.formats[amber_item] = cmap_tmpl_mol.formats[item]

for dihedral in pmd_mol.dihedrals:
    if dihedral.type is None or dihedral.ignore_end: continue
    if isinstance(dihedral.type, DihedralTypeList):
        for dt in dihedral.type:
            if dt.scee:
                dt.scee = 1.200
            if dt.scnb:
                dt.scnb = 2.000
        # print(dihedral, dihedral.type, scee_values, scnb_values)
    else:
        if dihedral.type.scee:
            dihedral.type.scee = 1.200
        if dihedral.type.scnb:
            dihedral.type.scnb = 2.000
            
pmd_mol.save(str(output_file), overwrite=True)
print(f"save to {output_file}")
    
            
            
            
            
            
                
                
            
            
            
            
            
            
            

