import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from parna.utils import parse_xyz_file
from aimnet2calc import AIMNet2Calculator
import json
from pathlib import Path
import argparse
import numpy as np


def calculate_energy(file_list, charge, jitted_model_path, output_file):
    calculator = AIMNet2Calculator(str(jitted_model_path))
    energy_info = {}
    for idx, ff in enumerate(file_list):
        ff = Path(ff)
        print(f"Calculating NNP energy for {ff}")
        
        data = parse_xyz_file(ff)
        data["charge"] = np.array([1,]).reshape(1,) * charge
        results = calculator(data, forces=False, stress=False, hessian=False)
        energy_info[ff.stem] = float(results["energy"].item())

    with open(output_file, "w") as f:
        json.dump(energy_info, f, indent=4)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", type=str, nargs='+', required=True)
    parser.add_argument("--jitted-model-path", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--charge", type=int, default=0)
    args = parser.parse_args()

    calculate_energy(args.file_list, args.charge, args.jitted_model_path, args.output_file)