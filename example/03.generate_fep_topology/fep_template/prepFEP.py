import amberti
from amberti.workflow import equilibrate_systems
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("CONFIG", type=str, help="config file")
args = parser.parse_args()

with open(args.CONFIG, "r") as f:
    config = json.load(f)


equilibrate_systems(
    config=config,
    top_dir="./top",
    suffix = ""
)