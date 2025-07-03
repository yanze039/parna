import sys
sys.path.append("/home/gridsan/ywang3/Project/Capping/parna/")
import parna
import parna.construct as construct
from parna.extract import extract_backbone
from parna.utils import read_yaml, remove_ter
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="oligo parameterization")
parser.add_argument("--oligo", help="oligo pdb file")
parser.add_argument("--output-dir", help="output directory")
parser.add_argument("--noncanonical-residues", default=[], help="noncanonical residues", nargs="+", type=int)
parser.add_argument("--keep-hydrogen", action="store_true", help="keep hydrogen atoms", default=False)
args = parser.parse_args()


oligo_pdb = Path(args.oligo)
bk_output_dir = Path(args.output_dir)
bk_output_dir.mkdir(exist_ok=True)

noncanonical_residues = args.noncanonical_residues

extract_backbone(
    str(oligo_pdb), 
    output_dir=bk_output_dir, 
    noncanonical_residues=noncanonical_residues, 
    residues_without_phosphate=[], 
    keep_hydrogen=args.keep_hydrogen
)

