from pathlib import Path
import parmed as pmd
import os
from parna.logger import getLogger
from typing import List
import shutil
import copy


logger = getLogger("parna.parm")


DATA = Path(__file__).parent/"data"
TEMPLATE = Path(__file__).parent/"template"
LIB = Path(__file__).parent/"lib"
FRAG = Path(__file__).parent/"fragments"
FRAME = Path(__file__).parent/"local_frame"
FRCMOD = DATA


def check_atomtype_mismatch(parm, rst7):
    pmd_mol = pmd.load_file(str(parm), str(rst7))
    mismatched_atoms = {"O2":[], "OS":[]}
    for atom in pmd_mol.atoms:
        if atom.type == "O2":
            if len(atom.bond_partners) != 1:
                mismatched_atoms["O2"].append((atom.idx, atom.name))
        elif atom.type == "OS":
            if len(atom.bond_partners) != 2:
                mismatched_atoms["OS"].append((atom.idx, atom.name))
    return mismatched_atoms


def parameterize(oligoFile, proteinFile=None, external_libs=[], 
                 additional_frcmods=[], output_dir=None, n_cations=0, n_anions=0, 
                 prefix="complex", check_atomtypes=True, solvated=True, saveparm=True, addons:List[str]=[]):
    oligoFile = Path(oligoFile)
    
    leap_content = []
    frcmod = FRCMOD / "ol3_gaff2.frcmod"
    atom_type_def = FRCMOD / "atom_type_def.in"
    with open(atom_type_def, "r") as fp:
        leap_content.extend(fp.readlines())
    leap_content.append(f"loadAmberParams {(frcmod).resolve()}")
    if isinstance(additional_frcmods, str) or isinstance(additional_frcmods, Path):
        additional_frcmods = [additional_frcmods]
    for fd in additional_frcmods:
        leap_content.append(f"loadAmberParams {(Path(fd)).resolve()}")
    leap_content.append(f"source leaprc.protein.ff14SB")
    leap_content.append(f"source leaprc.RNA.OL3")
    leap_content.append(f"source leaprc.water.tip3p")
    residue_libs = list(LIB.glob("*.lib"))
    # internal libs
    for reslib in residue_libs:
        leap_content.append(f"loadoff {str(reslib.resolve())}")
    # external libs
    if isinstance(external_libs, str) or isinstance(external_libs, Path):
        external_libs = [external_libs]
    for extlib in external_libs:
        leap_content.append(f"loadoff {Path(extlib).resolve()}")
    
    for line in addons:
        leap_content.append(line)
        
    if proteinFile is not None:
        proteinFile = Path(proteinFile)
    if output_dir is None:
        if proteinFile is not None:
            output_dir = oligoFile.stem + "_" + proteinFile.stem
        else:
            output_dir = oligoFile.stem
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    
    leap_content.append(f"oligo = loadpdb {str(oligoFile.resolve())}")
    if proteinFile is not None:
        leap_content.append(f"protein = loadpdb {proteinFile.resolve()}")
        leap_content.append(f"complex = combine {{ oligo protein }}")
    else:
        leap_content.append(f"complex = oligo")
    
    if solvated:
        leap_content.append(f"solvatebox complex TIP3PBOX 15.0 0.75")
        leap_content.append(f"addionsrand complex Na+ 0")
        leap_content.append(f"addionsrand complex Cl- 0")
        leap_content.append(f"addionsrand complex Na+ {n_cations}")
        leap_content.append(f"addionsrand complex Cl- {n_anions}")
    leap_content.append(f"savepdb complex {prefix}.pdb")
    if saveparm:
        leap_content.append(f"saveAmberParm complex {prefix}.parm7 {prefix}.rst7")
    leap_content.append("quit")
    cwd = Path.cwd()
    os.chdir(output_dir)
    with open(f"{oligoFile.stem}.tleap.in", "w") as fp:
        fp.write("\n".join(leap_content))
    code = os.system(f"tleap -f {oligoFile.stem}.tleap.in")
    if code != 0:
        print(f"tleap failed for {oligoFile.stem}.tleap.in")
        raise RuntimeError(f"tleap failed for {oligoFile.stem}.tleap.in")
    os.chdir(cwd)

    parmFile = output_dir/(prefix+".parm7")
    rst7File = output_dir/(prefix+".rst7")
    if check_atomtypes:
        logger.info("Checking atomtypes")
        mismatched_atoms = check_atomtype_mismatch(parmFile, rst7File)
        error = False   
        for atomtype, atoms in mismatched_atoms.items():
            if len(atoms) > 1:
                error = True
                logger.warning(f"Atomtype {atomtype} mismatched: {mismatched_atoms[atomtype]}")
        if error:
            raise RuntimeError("Atomtype mismatched")
        else:
            logger.info("Atomtype check passed")


def alchemical_parameterize(oligoFile1, oligoFile2, proteinFile=None, external_libs=[], additional_frcmods=[], output_dir=None, n_cations=0, n_anions=0, prefix="complex", check_atomtypes=True):
    oligoFile1 = Path(oligoFile1)
    oligoFile2 = Path(oligoFile2)
    
    leap_content = []
    frcmod = FRCMOD / "ol3_gaff2.frcmod"
    leap_content.append(f"loadAmberParams {(frcmod).resolve()}")
    if isinstance(additional_frcmods, str) or isinstance(additional_frcmods, Path):
        additional_frcmods = [additional_frcmods]
    additional_frcmods.append(str(Path(__file__).parent / "data" / "frcmod.99bsc0-chiol3-CaseP.frcmod"))
    for fd in additional_frcmods:
        leap_content.append(f"loadAmberParams {(Path(fd)).resolve()}")
    leap_content.append(f"source leaprc.protein.ff14SB")
    leap_content.append(f"source leaprc.RNA.OL3")
    leap_content.append(f"source leaprc.water.tip3p")
    residue_libs = list(LIB.glob("*.lib"))
    # internal libs
    for reslib in residue_libs:
        leap_content.append(f"loadoff {str(reslib.resolve())}")
    # external libs
    if isinstance(external_libs, str) or isinstance(external_libs, Path):
        external_libs = [external_libs]
    for extlib in external_libs:
        leap_content.append(f"loadoff {Path(extlib).resolve()}")
    if proteinFile is not None:
        proteinFile = Path(proteinFile)
    if output_dir is None:
        if proteinFile is not None:
            output_dir = oligoFile1.stem + "_" + oligoFile2.stem + "_" + proteinFile.stem
        else:
            output_dir = oligoFile1.stem + "_" + oligoFile2.stem
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    
    leap_content.append(f"oligo1 = loadpdb {str(oligoFile1.resolve())}")
    leap_content.append(f"oligo2 = loadpdb {str(oligoFile2.resolve())}")
    if proteinFile is not None:
        leap_content.append(f"protein = loadpdb {proteinFile.resolve()}")
        leap_content.append(f"complex = combine {{ oligo1 oligo2 protein }}")
    else:
        leap_content.append(f"complex = combine {{ oligo1 oligo2 }}")
    leap_content.append(f"center complex")
    leap_content.append(f"translate complex {{13.0, 13.0, 13.0}}")
    leap_content.append(f"solvatebox complex TIP3PBOX 15.0 0.75")
    leap_content.append(f"addionsrand complex Na+ {n_cations}")
    leap_content.append(f"addionsrand complex Cl- {n_anions}")
    leap_content.append(f"savepdb complex {prefix}.pdb")
    leap_content.append(f"saveAmberParm complex {prefix}.parm7 {prefix}.rst7")
    leap_content.append("quit")
    cwd = Path.cwd()
    os.chdir(output_dir)
    logger.info("Change to output directory: %s", output_dir)
    with open(f"{output_dir.stem}.tleap.in", "w") as fp:
        fp.write("\n".join(leap_content))
    code = os.system(f"tleap -f {output_dir.stem}.tleap.in")
    if code != 0:
        logger.warning(f"tleap failed for {output_dir.stem}.tleap.in")
        raise RuntimeError(f"tleap failed for {output_dir.stem}.tleap.in")
    else:
        logger.info("Done!")
    os.chdir(cwd)

    parmFile = output_dir/(prefix+".parm7")
    rst7File = output_dir/(prefix+".rst7")
    if check_atomtypes:
        logger.info("Checking atomtypes")
        mismatched_atoms = check_atomtype_mismatch(parmFile, rst7File)
        error = False   
        for atomtype, atoms in mismatched_atoms.items():
            if len(atoms) > 1:
                error = True
                logger.warning(f"Atomtype {atomtype} mismatched: {mismatched_atoms[atomtype]}")
        if error:
            raise RuntimeError("Atomtype mismatched")
        else:
            logger.info("Atomtype check passed")


def get_residue_template_container(pmd_mol):
    if (type(pmd_mol) == pmd.modeller.residue.ResidueTemplateContainer):
        container = pmd_mol
    elif (type(pmd_mol) == pmd.structure.Structure):
        container = pmd.modeller.residue.ResidueTemplateContainer().from_structure(pmd_mol)
    elif (type(pmd_mol) == pmd.modeller.residue.ResidueTemplate):
        container = pmd.modeller.residue.ResidueTemplateContainer().from_library({pmd_mol.name: pmd_mol})
    else:
        raise ValueError("Unknown type")
    return container


def generate_frcmod(input_file, output_file, major_forcefield="ol3", 
                    minor_forcefield="gaff2", parm_set="parm10", atom_type="amber", 
                    sinitize=True, clean=True, output_mol2=None):  
    _mol2_file = Path(output_file).parent / (Path(input_file).stem + ".tmp.mol2") 
    if (not Path(input_file).suffix == ".mol2") or sinitize:
        
        command_antechamber = [
            "antechamber",
            "-fi", Path(input_file).suffix[1:],
            "-i", str(input_file),
            "-fo", "mol2",
            "-o", str(_mol2_file),
            "-at", atom_type,
            "-pf", "y"
        ]
        logger.info(" ".join(command_antechamber))
        code = os.system(" ".join(command_antechamber))
        if code != 0:
            logger.error("Antechamber failed")
            raise RuntimeError("Antechamber failed")
    else:
        shutil.copy(input_file, _mol2_file)
    
    # check if any DU atom exists in the mol2 file
    _tmp_mol = pmd.load_file(str(_mol2_file))
    tmp_Structure = pmd.Structure()
    container = get_residue_template_container(_tmp_mol)
    _mol2_file_2 = None
    for idx, residue in enumerate(container):
        for atom in residue.atoms:
            # BUG: antechamber does not handle DU atom correctly
            # handle DU atom and NO atom.
            # DU atoms are atom types not defined in Amber forcefield
            # NO atoms can be assigned by antechamber, 
            # but it actually does not exist in PARM10.
            if atom.type == "DU" or atom.type == "NO" or atom.type == "N1":
                logger.warning(f"DU atom found in mol2 file {atom}")
                logger.warning("Replacing DU with atom type from gaff2")
                _mol2_file_2 = Path(output_file).parent / (Path(input_file).stem + ".tmp2.mol2") 
                if not os.path.exists(_mol2_file_2):
                    command_antechamber = [
                        "antechamber",
                        "-fi", Path(input_file).suffix[1:],
                        "-i", str(input_file),
                        "-fo", "mol2",
                        "-o", str(_mol2_file_2),
                        "-at", "gaff2",
                        "-pf", "y"
                    ]
                    logger.info(" ".join(command_antechamber))
                    code = os.system(" ".join(command_antechamber))
                    if code != 0:
                        logger.error("Antechamber failed")
                        raise RuntimeError("Antechamber failed")
                _tmp_mol_container_2 = get_residue_template_container(pmd.load_file(str(_mol2_file_2)))
                atom.type = _tmp_mol_container_2[idx].map[atom.name].type
            tmp_Structure.add_atom(
                copy.deepcopy(atom), 
                resname=residue.name,
                resnum=idx+1,
                chain="A"
            )
    tmp_Structure.assign_bonds({r.name: r for r in container})
    tmp_Structure.save(str(_mol2_file), overwrite=True)
    if output_mol2 is not None:
        shutil.copy(_mol2_file, output_mol2)
    
    _output_file = Path(output_file).parent / ("_" + Path(output_file).stem + ".frcmod")
    command_major = [
        "parmchk2",
        "-i", str(_mol2_file),
        "-f", "mol2",
        "-o", str(_output_file),
        "-a", "Y",
        "-s", parm_set,
        "-frc", major_forcefield
    ]
    logger.info(" ".join(command_major))
    code = os.system(" ".join(command_major))
    if code != 0:
        logger.error("Parmchk2 failed")
        raise RuntimeError("Parmchk2 failed")
    command_minor = [
        "parmchk2",
        "-i", str(_output_file),
        "-f", "frcmod",
        "-o", str(output_file),
        "-a", "Y",
        "-s", minor_forcefield,
        "-frc", major_forcefield,
        "-att", "2"
    ]
    logger.info(" ".join(command_minor))
    os.system(" ".join(command_minor))
    if clean:
        os.remove(_mol2_file)
        os.remove(_output_file)
        if _mol2_file_2 is not None:
            if os.path.exists(_mol2_file_2):
                os.remove(_mol2_file_2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbFile", type=str, help="pdb file to be parameterized")
    parser.add_argument("--external_libs", type=str, nargs="*", help="external lib files")
    parser.add_argument("--protein", type=str, help="protein file")
    args = parser.parse_args()
    parameterize(args.pdbFile, args.protein, args.external_libs)


