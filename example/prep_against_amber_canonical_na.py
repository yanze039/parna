import parna
import parna.resp as resp
import parna.construct as construct
from parna.extract import extract_backbone, extract_residue_from_pdb
from parna.construct import build_residue_pdb_block, replace_residue
from parna.parm import alchemical_parameterize, parameterize, generate_frcmod
from parna.fep import get_softcore_region
from parna.utils import list2string
import rdkit
import rdkit.Chem as Chem
from pathlib import Path
import yaml
import json
import parmed
import argparse
import os
import shutil

def prep_fep_config(
        timask1, timask2, scmask1, scmask2,
        output_file
    ):
    template="/home/gridsan/ywang3/Project/Capping/oligo_simulation/test/example/config.json"
    with open(template, "r") as f:
        config = json.load(f)
    config["prep"]["timask1"] = timask1
    config["prep"]["timask2"] = timask2
    config["prep"]["scmask1"] = scmask1
    config["prep"]["scmask2"] = scmask2
    config["unified"]["timask1"] = timask1
    config["unified"]["timask2"] = timask2
    config["unified"]["scmask1"] = scmask1
    config["unified"]["scmask2"] = scmask2
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)


def oligoprep(yaml_file):
    config_file = Path(yaml_file)
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input_pdb = Path(config["input_pdb"])
    n_conformers = int(config["n_conformers"])
    charge_of_block = int(config["charge"])
    resname = str(config["resname"])
    skip_constraint = config["skip_constraint"]
    output_dir = Path(config.get("lib_output_dir", f"{resname}/{resname}.lib"))
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    # make parameters for the input file.
        
    resp.RESP(
        input_file=input_pdb,
        charge=charge_of_block,
        output_dir=output_dir,
        residue_name=resname,
        skip_constraint=skip_constraint,
        n_conformers=n_conformers,
        memory=config["psi4_memory"], 
        n_threads=config["psi4_threads"], 
        method_basis=config["method_basis"],
    )

    # make fragments/residue for the input file.
    input_mol2_file = f"{output_dir}/{input_pdb.stem}.mol2"
    construct.make_fragment(
        input_file=input_mol2_file,
        residue_name=resname,
        residue_type=config["residue_type"],
        output_dir=output_dir,
        charge=charge_of_block
    )

    # extract backbone from the input file.
    oligo_pdb = Path(config["oligo_template"])
    bk_output_dir = Path(config["extract_output_dir"])
    noncanonical_residues = list(config["noncanonical_residues"])

    extract_backbone(
            str(oligo_pdb), 
            output_dir=bk_output_dir, 
            noncanonical_residues=noncanonical_residues, 
            residues_without_phosphate=[], 
            keep_hydrogen=False
        )

    # flexible aligning
    input_mol2_file = output_dir / f"{resname}.mol2"
    
    tmpl_oligo = parmed.load_file(str(oligo_pdb))
    residue_position = config["residue_position"]
    template_file = bk_output_dir/f"{tmpl_oligo.residues[residue_position-1].name}_{residue_position}.pdb"

    pdb_block = build_residue_pdb_block(
        input_file=input_mol2_file, 
        residue_name=resname, 
        template_residue=template_file, 
        atomtype="amber"
    )
    with open(bk_output_dir/f"{resname}.pdb", "w") as f:
        f.write(pdb_block)

    # change residue
    replace_residue(
        oligo_pdb,
        bk_output_dir/f"{resname}.pdb",
        residue_position,
        bk_output_dir/f"{oligo_pdb.stem}.mod.{resname}.pdb"
    )
    
    oligo_file_1 = Path(config["oligo_template"])
    oligo_file_2 = bk_output_dir/f"{oligo_pdb.stem}.mod.{resname}.pdb"

    if isinstance(config["external_libs"], list):
        external_libs = config["external_libs"] + [output_dir/f"{resname}.lib"]
    else:
        external_libs = [config["external_libs"], output_dir/f"{resname}.lib"]
    
    fep_dir = Path(config.get("fep_dir", f"{resname}/FEP.{resname}.run"))
    topology_dir = fep_dir / "top"

    generate_frcmod(
        input_file=oligo_file_2,
        output_file=output_dir/f"{resname}.frcmod",
    )

    # for complex-leg
    alchemical_parameterize(
        oligoFile1=oligo_file_1, 
        oligoFile2=oligo_file_2, 
        proteinFile=config["protein_pdb"], 
        external_libs=external_libs, 
        additional_frcmods=output_dir/f"{resname}.frcmod",
        output_dir=topology_dir, 
        n_cations=config["n_cations_complex"],
        n_anions=config["n_anions_complex"],
        prefix="complex", 
        check_atomtypes=True 
    )

    # for ligands-leg
    alchemical_parameterize(
        oligoFile1=oligo_file_1, 
        oligoFile2=oligo_file_2, 
        proteinFile=None, 
        external_libs=external_libs, 
        additional_frcmods=output_dir/f"{resname}.frcmod",
        output_dir=topology_dir, 
        n_cations=config["n_cations_ligands"],
        n_anions=config["n_anions_ligands"], 
        prefix="ligands",
        check_atomtypes=True 
    )
    
    # for protein-mod complex
    parameterize(
        oligoFile=oligo_file_2, 
        proteinFile=config["protein_pdb"], 
        external_libs=external_libs, 
        additional_frcmods=output_dir/f"{resname}.frcmod",
        output_dir=topology_dir, 
        n_cations=config["n_cations_complex"],
        n_anions=config["n_anions_complex"], 
        prefix=f"{resname}_protein", 
        check_atomtypes=True 
    )
    
    parameterize(
        oligoFile=str(oligo_file_1),
        external_libs=external_libs, 
        additional_frcmods=str(output_dir/f"{resname}.frcmod"),
        output_dir=topology_dir,
        prefix=f"{oligo_file_1.stem}.addH", 
        solvated=False,
        saveparm=False,
        check_atomtypes=False
    )
    extract_residue_from_pdb(
        input_file=topology_dir/f"{oligo_file_1.stem}.addH.pdb", 
        output_file=topology_dir/f"{tmpl_oligo.residues[residue_position-1].name}.template.pdb", 
        residue_number=residue_position
    )

    template_residue = Chem.MolFromPDBFile(str(topology_dir/f"{tmpl_oligo.residues[residue_position-1].name}.template.pdb"), removeHs=False)
    query_residue = Chem.MolFromPDBFile(str(bk_output_dir/f"{resname}.pdb"), removeHs=False)
    query_unique_atom_names, template_unique_atom_names = get_softcore_region(query_residue, template_residue, return_atom_name=True)

    n_residues = len(tmpl_oligo.residues)
    timask1 = f":1-{n_residues:d}"
    timask2 = f":{n_residues+1:d}-{n_residues*2:d}"
    if len(template_unique_atom_names) > 0:
        scmask1 = f":{residue_position}@{list2string(template_unique_atom_names)}"
    else:
        scmask1 = ""
    if len(query_unique_atom_names) > 0:
        scmask2 = f":{residue_position+n_residues}@{list2string(query_unique_atom_names)}"
    else:
        scmask2 = ""

    print(f"timask1: {timask1}")
    print(f"timask2: {timask2}")
    print(f"scmask1: {scmask1}")
    print(f"scmask2: {scmask2}")
    prep_fep_config(
        timask1, timask2, scmask1, scmask2,
        output_file=fep_dir/"config.json"
    )
    prep_script = Path("example/prepFEP.py")
    run_script = Path("example/runFEP.py")
    shutil.copy(prep_script, fep_dir/Path(prep_script.name))
    shutil.copy(run_script, fep_dir/Path(run_script.name))
    print("File saved to ", fep_dir)

    # submit job:
    with open(fep_dir/"submit.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("source /etc/profile\n")
        f.write("source $HOME/env/amber22.env\n")
        f.write(f"cd {fep_dir.resolve()}\n")
        f.write("python prepFEP.py config.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oligo parameterization")
    parser.add_argument("yaml_file", help="yaml file for oligo parameterization")
    args = parser.parse_args()
    oligoprep(args.yaml_file)







