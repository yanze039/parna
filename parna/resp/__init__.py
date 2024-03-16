from parna.resp.conformer import gen_conformer
from parna.resp.psi4 import calculate_energy
from parna.resp.fitting import fit_charges
from pathlib import Path
from parna.logger import getLogger

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
        method_basis="HF/6-31G*"
    ):
    logger.info(f"Generating conformers for {input_file}...")
    gen_conformer(
        query_file=input_file,
        scan_steps=n_conformers,
        charge=charge,
        output_dir=output_dir,
        skip_constraint=skip_constraint
    )
    conformer_files = list(Path(output_dir).glob("conformer_*.xyz"))
    for conformer_file in conformer_files:
        logger.info(f"Calculating energy for {conformer_file}")
        calculate_energy(
            conformer_file,
            output_dir, 
            charge=charge, 
            memory=memory,
            n_threads=n_threads,
            method_basis=method_basis
        )
    logger.info("Fitting charges...")
    fit_charges(
        input_file=input_file,
        wfn_directory=output_dir, 
        output_dir=output_dir, 
        residue_name=residue_name, 
        tightness=0.1
    )
    logger.info("RESP calculation finished.")
    return None



