from pathlib import Path
from propose.datasets.human36m.loaders import pickle_poses

input_dir = Path("/data/human36m/raw/")
output_dir = Path("/data/human36m/processed/")

print("Loading data...")
pickle_poses(input_dir, output_dir)
print("Done.")
