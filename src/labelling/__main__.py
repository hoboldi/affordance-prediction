import argparse
from pathlib import Path
from .server import run

parser = argparse.ArgumentParser(description="Manual affordance labelling tool")
parser.add_argument("--data-root", default="data", type=Path, metavar="DIR")
parser.add_argument("--port", default=8765, type=int)
parser.add_argument("--split", default=None, choices=["train", "val"])
args = parser.parse_args()

run(args.data_root.resolve(), port=args.port, split=args.split)
