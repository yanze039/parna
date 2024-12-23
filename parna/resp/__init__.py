from parna.resp.conformer import gen_conformer
from parna.qm import calculate_energy
from parna.resp.fitting import fit_charges
from parna.resp.cap import fit_charges_cap
from parna.resp.frag import fit_charges_frag
from pathlib import Path
from parna.logger import getLogger
from parna.qm.xtb_utils import structure_optimization, gen_multi_conformations
from parna.utils import map_atoms, rd_load_file, split_xyz_file
import parmed as pmd
import os
import numpy as np
import shutil
from typing import List, Union

logger = getLogger(__name__)


def generate_atomic_charges(
        input_files: Union[str, List[str]],
        charge,
        output_dir,
        residue_name,
        scheme="resp2",
        aqueous_ratio=0.6,
        nucleoside=False,
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31G*",
        charge_constrained_groups=["OH5", "OH3"],
        extra_charge_constraints={},
        extra_equivalence_constraints=[],
        aqueous=False,
        overwrite=False,
        prefix=None,
        generate_conformers=False,
        skip_constraint=None,
        n_conformers=6
    ):
    if scheme == "resp":
        _generate_atomic_charges(
            input_files=input_files,
            charge=charge,
            output_dir=output_dir,
            residue_name=residue_name,
            nucleoside=nucleoside,
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis,
            extra_charge_constraints=extra_charge_constraints,
            extra_equivalence_constraints=extra_equivalence_constraints,
            aqueous=aqueous,
            overwrite=overwrite,
            prefix=prefix,
            generate_conformers=generate_conformers,
            skip_constraint=skip_constraint,
            charge_constrained_groups=charge_constrained_groups,
            n_conformers=n_conformers
        )
    elif scheme == "resp2":
        # generate atomic charges for aqueous phase
        aq_output_dir = Path(output_dir)/"resp2_aqueous"
        aq_prefix = "resp2_aq"
        _generate_atomic_charges(
            input_files=input_files,
            charge=charge,
            output_dir=aq_output_dir,
            residue_name=residue_name,
            nucleoside=nucleoside,
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis,
            extra_charge_constraints=extra_charge_constraints,
            extra_equivalence_constraints=extra_equivalence_constraints,
            aqueous=True,
            overwrite=overwrite,
            prefix=aq_prefix,
            generate_conformers=generate_conformers,
            skip_constraint=skip_constraint,
            charge_constrained_groups=charge_constrained_groups,
            n_conformers=n_conformers
        )
        # generate atomic charges for gas phase
        gas_output_dir = Path(output_dir)/"resp2_gas"
        gas_prefix = "resp2_gas"
        _generate_atomic_charges(
            input_files=input_files,
            charge=charge,
            output_dir=gas_output_dir,
            residue_name=residue_name,
            nucleoside=nucleoside,
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis,
            extra_charge_constraints=extra_charge_constraints,
            extra_equivalence_constraints=extra_equivalence_constraints,
            aqueous=False,
            overwrite=overwrite,
            prefix=gas_prefix,
            generate_conformers=generate_conformers,
            skip_constraint=skip_constraint,
            charge_constrained_groups=charge_constrained_groups,
            n_conformers=n_conformers
        )
        aq_charged_mol2 = aq_output_dir/f"{aq_prefix}.mol2"
        gas_charged_mol2 = gas_output_dir/f"{gas_prefix}.mol2"
        # combine charges
        if prefix is None:
            prefix = residue_name + ".resp2"
        resp2_charged_mol2 = Path(output_dir)/f"{prefix}.mol2"
        shutil.copy(aq_charged_mol2, resp2_charged_mol2)
        pmd_gas = pmd.load_file(str(gas_charged_mol2))
        pmd_aq = pmd.load_file(str(aq_charged_mol2))
        pmd_resp2 = pmd.load_file(str(resp2_charged_mol2))
        for atom in pmd_resp2.atoms:
            atom.charge = (1-aqueous_ratio)*pmd_gas[atom.idx].charge + aqueous_ratio*pmd_aq[atom.idx].charge
        pmd_resp2.save(str(resp2_charged_mol2), overwrite=True)
        



def _generate_atomic_charges(
        input_files: Union[str, List[str]],
        charge,
        output_dir,
        residue_name,
        nucleoside=False,
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31G*",
        extra_charge_constraints={},
        extra_equivalence_constraints=[],
        aqueous=False,
        overwrite=False,
        n_conformers=6,
        prefix=None,
        generate_conformers=False,
        skip_constraint=None,
        charge_constrained_groups=["OH5", "OH3"]
    ):
    
    if nucleoside:
        RESP(
            input_file=input_files,
            charge=charge,
            output_dir=output_dir,
            residue_name=residue_name,
            memory=memory, 
            n_threads=n_threads, 
            method_basis=method_basis,
            n_conformers=n_conformers,
            aqueous=aqueous,
            prefix=prefix,
            skip_constraint=skip_constraint,
            charge_constrained_groups=charge_constrained_groups,
            engine="orca"
        )
    else:
        if generate_conformers and len(input_files) == 1:
            RESP_molecule(
                input_file=input_files[0],
                charge=charge,
                output_dir=output_dir,
                residue_name=residue_name,
                memory=memory, 
                n_threads=n_threads, 
                method_basis=method_basis,
                n_conformers=n_conformers,
                aqueous=aqueous
            )
        else:
            RESP_fragment(
                input_files=input_files,
                charge=charge,
                output_dir=output_dir,
                residue_name=residue_name,
                memory=memory, 
                n_threads=n_threads, 
                method_basis=method_basis,
                extra_charge_constraints=extra_charge_constraints,
                extra_equivalence_constraints=extra_equivalence_constraints,
                aqueous=aqueous,
                overwrite=overwrite,
                prefix=prefix
            )
    
    pass


def RESP(
        input_file,
        charge,
        output_dir,
        residue_name,
        skip_constraint=None,
        n_conformers=6,
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31G*",
        aqueous=False,
        engine="psi4",
        charge_constrained_groups=["OH5", "OH3"],
        prefix=None
    ):
    logger.info(f"Generating conformers for {input_file}...")
    gen_conformer(
        query_file=input_file,
        scan_steps=n_conformers,
        charge=charge,
        output_dir=output_dir,
        skip_constraint=skip_constraint,
        prefix="resp_conformer"
    )
    conformer_files = list(Path(output_dir).glob("resp_conformer_*.xyz"))
    for conformer_file in conformer_files:
        logger.info(f"Calculating energy for {conformer_file}")
        if os.path.exists(f"{output_dir}/{(conformer_file).stem}.psi4.fchk"):
            continue
        calculate_energy(
            conformer_file,
            output_dir, 
            charge=charge, 
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis,
            aqueous=aqueous,
            engine=engine
        )
    logger.info("Fitting charges...")
    if engine == "orca":
        wfn_file_type = "molden"
    else:
        wfn_file_type = "fchk"
    fit_charges(
        input_file=input_file,
        wfn_directory=output_dir, 
        output_dir=output_dir, 
        residue_name=residue_name, 
        tightness=0.1,
        wfn_file_type=wfn_file_type,
        charge_constrained_groups=charge_constrained_groups,
        prefix=prefix
    )
    logger.info("RESP calculation finished.")
    return None



# def RESP_cap(
#         input_file,
#         charge,
#         output_dir,
#         residue_name,
#         memory="160 GB", 
#         n_threads=48, 
#         method_basis="HF/6-31+G*",
#         restraint_template=None
#     ):
#     TEMPLATE = Path(__file__).parent.parent/"template"
#     logger.info(f"Generating conformers for {input_file}...")
#     output_dir = Path(output_dir)

#     # add restraints on heavy atoms
#     if restraint_template is None:
#         cap_template = str(TEMPLATE/"m7gppp.pdb")    
    
#     template = rd_load_file(cap_template, removeHs=False)
#     query = rd_load_file(input_file, removeHs=False)
#     atom_map = map_atoms(
#         template, 
#         query
#     )
#     restrained_atom_list = []
#     for atom_pair in atom_map:
#         atomic_number = query.GetAtomWithIdx(atom_pair[1]).GetAtomicNum()
#         if atomic_number != 1:  # only add heavy atoms
#             restrained_atom_list.append(atom_pair[1]+1)
#     # opt hydrogens and new groups.
#     structure_optimization(input_file, charge, restrained_atom_list, workdir=str(output_dir))
    
#     conf_file = str(output_dir/f"xtbopt.pdb")
#     calculate_energy(
#         conf_file,
#         str(output_dir), 
#         charge=charge, 
#         memory=memory,
#         n_threads=n_threads,
#         method_basis=method_basis
#     )

#     fit_charges_cap(
#         input_file=conf_file,
#         wfn_file=str(output_dir/f"xtbopt.psi4.fchk"), 
#         output_dir=str(output_dir), 
#         residue_name=residue_name, 
#         tightness=0.1
#     )
#     os.rename(output_dir/f"xtbopt.0.100.mol2", output_dir/f"{Path(input_file).stem}.mol2")
    

def RESP_fragment(
        input_files: Union[str, List[str]],
        charge,
        output_dir,
        residue_name,
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31G*",
        extra_charge_constraints={},
        extra_equivalence_constraints=[],
        aqueous=False,
        overwrite=False,
        prefix=None
    ):
    logger.info(f"RESP charging {input_files}...")
    output_dir = Path(output_dir)  
    output_dir.mkdir(exist_ok=True)
    
    if type(input_files) == str:
        conformers = [input_files]
    else:
        conformers = input_files
    
    for conformer_file in conformers:
        if (os.path.exists(f"{output_dir}/{(Path(conformer_file)).stem}.psi4.fchk") \
            or os.path.exists(f"{output_dir}/{(Path(conformer_file)).stem}.molden")) \
                and not overwrite:
            continue
        calculate_energy(
            conformer_file,
            str(output_dir), 
            charge=charge, 
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis,
            aqueous=aqueous,
            engine="orca"
        )

    fit_charges_frag(
        input_file=conformers[0],
        wfn_directory=str(output_dir), 
        output_dir=str(output_dir), 
        residue_name=residue_name, 
        tightness=0.1,
        wfn_file_type="molden",
        extra_charge_constraints=extra_charge_constraints,
        extra_equivalence_constraints=extra_equivalence_constraints,
        prefix=prefix
    )


def RESP_molecule(
        input_file,
        charge,
        output_dir,
        residue_name,
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31G*",
        n_conformers=6,
        aqueous=False
    ):
    logger.info(f"Generating conformers for {input_file}...")
    output_dir = Path(output_dir)  
    
    gen_multi_conformations(
        input_file, 
        charge, 
        output_dir, 
        ewin=6, 
        threads=48
    )
    xtb_energies = np.loadtxt(output_dir/"crest.energies")
    n_xtb_conf = len(xtb_energies)
    if n_xtb_conf <= n_conformers:
        n_conformers = n_xtb_conf
        every_frame = 1
    else:
        every_frame = n_xtb_conf//n_conformers
    
    split_xyz_file(
        output_dir/"crest_conformers.xyz", 
        output_folder=output_dir, 
        every_frame=every_frame,
        output_prefix="resp_conformer"
    )
    
    conformers = list(output_dir.glob("resp_conformer_*.xyz"))
    
    for conformer_file in conformers:
        calculate_energy(
            conformer_file,
            str(output_dir), 
            charge=charge, 
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis,
            aqueous=aqueous,
        )

    fit_charges_frag(
        input_file=input_file,
        wfn_directory=str(output_dir), 
        output_dir=str(output_dir), 
        residue_name=residue_name, 
        tightness=0.1,
        wfn_file_type="fchk"
    )
    return None