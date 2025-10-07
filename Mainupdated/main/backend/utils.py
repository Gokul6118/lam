import os
from typing import List

def cleanup_temp_files(file_paths: List[str]):
    """
    Safely removes a list of temporary files.
    """
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {path}: {e}")