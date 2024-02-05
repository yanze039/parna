from pathlib import Path
import parmed as pmd
import os
from parna.logger import getLogger


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


def parameterize(oligoFile, proteinFile=None, external_libs=[], output_dir=None, n_ions=50, prefix="complex", check_atomtypes=True):
    oligoFile = Path(oligoFile)
    
    leap_content = []
    frcmod = FRCMOD / "ol3_gaff2.frcmod"
    leap_content.append(f"source leaprc.protein.ff14SB")
    leap_content.append(f"source leaprc.RNA.OL3")
    leap_content.append(f"source leaprc.water.tip3p")
    leap_content.append(f"loadAmberParams {(frcmod).resolve()}")
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
    leap_content.append(f"solvatebox complex TIP3PBOX 15.0 0.75")
    leap_content.append(f"addionsrand complex Na+ 0")
    leap_content.append(f"addionsrand complex Cl- 0")
    leap_content.append(f"addionsrand complex Na+ {n_ions}")
    leap_content.append(f"addionsrand complex Cl- {n_ions}")
    leap_content.append(f"savepdb complex {prefix}.pdb")
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


def alchemical_parameterize(oligoFile1, oligoFile2, proteinFile=None, external_libs=[], output_dir=None, n_ions=50, prefix="complex", check_atomtypes=True):
    oligoFile1 = Path(oligoFile1)
    oligoFile2 = Path(oligoFile2)
    
    leap_content = []
    frcmod = FRCMOD / "ol3_gaff2.frcmod"
    leap_content.append(f"source leaprc.protein.ff14SB")
    leap_content.append(f"source leaprc.RNA.OL3")
    leap_content.append(f"source leaprc.water.tip3p")
    leap_content.append(f"loadAmberParams {(frcmod).resolve()}")
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
    leap_content.append(f"solvatebox complex TIP3PBOX 15.0 0.75")
    leap_content.append(f"addionsrand complex Na+ 0")
    leap_content.append(f"addionsrand complex Cl- 0")
    leap_content.append(f"addionsrand complex Na+ {n_ions}")
    leap_content.append(f"addionsrand complex Cl- {n_ions}")
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbFile", type=str, help="pdb file to be parameterized")
    parser.add_argument("--external_libs", type=str, nargs="*", help="external lib files")
    parser.add_argument("--protein", type=str, help="protein file")
    args = parser.parse_args()
    parameterize(args.pdbFile, args.protein, args.external_libs)


