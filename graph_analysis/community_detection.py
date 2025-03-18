# Louvain algorithm
import os
from pathlib import Path

file_dir = Path(__file__).parent
lastfm_data_path = file_dir.parent / "lastfm_asia"
deezer_data_path = file_dir.parent / "deezer_europe"

print(os.path.isdir(deezer_data_path))

assert os.path.isdir(
    deezer_data_path
), f"Deezer data path is missing: {deezer_data_path}"
assert os.path.isdir(
    lastfm_data_path
), f"LastFM data path is missing: {lastfm_data_path}"
