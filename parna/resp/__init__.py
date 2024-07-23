from parna.resp.conformer import gen_conformer
from parna.qm import calculate_energy
from parna.resp.fitting import fit_charges
from parna.resp.cap import fit_charges_cap
from parna.resp.frag import fit_charges_frag
from pathlib import Path
from parna.logger import getLogger
from parna.qm.xtb import structure_optimization, gen_multi_conformations
from parna.utils import map_atoms, rd_load_file, split_xyz_file
import os
import numpy as np

logger = getLogger(__name__)


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
        engine="psi4"
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
        wfn_file_type=wfn_file_type
    )
    logger.info("RESP calculation finished.")
    return None



def RESP_cap(
        input_file,
        charge,
        output_dir,
        residue_name,
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31+G*",
        restraint_template=None
    ):
    TEMPLATE = Path(__file__).parent.parent/"template"
    logger.info(f"Generating conformers for {input_file}...")
    output_dir = Path(output_dir)

    # add restraints on heavy atoms
    if restraint_template is None:
        cap_template = str(TEMPLATE/"m7gppp.pdb")    
    
    template = rd_load_file(cap_template, removeHs=False)
    query = rd_load_file(input_file, removeHs=False)
    atom_map = map_atoms(
        template, 
        query
    )
    restrained_atom_list = []
    for atom_pair in atom_map:
        atomic_number = query.GetAtomWithIdx(atom_pair[1]).GetAtomicNum()
        if atomic_number != 1:  # only add heavy atoms
            restrained_atom_list.append(atom_pair[1]+1)
    # opt hydrogens and new groups.
    structure_optimization(input_file, charge, restrained_atom_list, workdir=str(output_dir))
    
    conf_file = str(output_dir/f"xtbopt.pdb")
    calculate_energy(
        conf_file,
        str(output_dir), 
        charge=charge, 
        memory=memory,
        n_threads=n_threads,
        method_basis=method_basis
    )

    fit_charges_cap(
        input_file=conf_file,
        wfn_file=str(output_dir/f"xtbopt.psi4.fchk"), 
        output_dir=str(output_dir), 
        residue_name=residue_name, 
        tightness=0.1
    )
    os.rename(output_dir/f"xtbopt.0.100.mol2", output_dir/f"{Path(input_file).stem}.mol2")
    

def RESP_fragment(
        input_file,
        charge,
        output_dir,
        residue_name,
        memory="160 GB", 
        n_threads=48, 
        method_basis="HF/6-31G*",
        n_conformers=6
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
            method_basis=method_basis
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