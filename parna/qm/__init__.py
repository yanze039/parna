from parna.qm.psi4_utils import calculate_energy_psi4
from parna.qm.orca_utils import calculate_energy_orca

def calculate_energy(input_file,
                     output_dir, 
                     charge=0, 
                     memory="160 GB", 
                     n_threads=48, 
                     method_basis="HF/6-31G*",
                     aqueous=False,
                     area=0.3,
                     engine="psi4"
    ):
    if engine == "orca":
        code = calculate_energy_orca(
            input_file,
            output_dir, 
            charge=charge, 
            n_threads=n_threads, 
            method_basis=method_basis,
            aqueous=aqueous
        )
    elif engine == "psi4":
        code = calculate_energy_psi4(
            input_file,
            output_dir, 
            charge=charge, 
            memory=memory, 
            n_threads=n_threads, 
            method_basis=method_basis,
            aqueous=aqueous,
            area=area,
            script_run=False
        )
    else:
        raise ValueError(f"Engine {engine} is not supported")
    return code

