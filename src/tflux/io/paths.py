import os
from datetime import datetime
from pathlib import Path


def find_root() -> Path:
    """
    project_root is inferred as the directory containing both 'src' and 'data'.
    """
    # --- infer project root from this file's location ---
    here = Path(__file__).resolve()

    # Walk upward until we find a folder that has both 'src' and 'data'
    project_root = None
    for p in [here.parent, *here.parents]:
        if (p / "src").is_dir() and (p / "data").is_dir():
            project_root = p
            break
    if project_root is None:
        # Fallback: assume repo root is two levels above src/tflux/...
        project_root = here.parents[2]

    return project_root


def get_default_data_dir() -> Path:

    root_path: Path = find_root()

        # --- default data directory ---
    data_dir = root_path / "data" / "raw" / "temp" / "WT"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    return data_dir


def make_output_dir(subdir_list: list[str]) -> Path:

    root_path: Path = find_root()

    outputs_dir = root_path / "outputs" 

    # If tflux/outputs doesn't exist
    outputs_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')

    run_number = 1

    # Find next available run number
    while True:
        dir_name = f"{date_str}_{run_number:03d}"
        new_dir = outputs_dir / dir_name
        
        if not new_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=True)
            for subdir in subdir_list:
                (new_dir / subdir).mkdir()
            return new_dir
        
        run_number += 1


def prepare_io(set_data_dir_path: Path = None, set_output_dir_path: Path = None, include_root = False) -> Path:

    data_dir_path = set_data_dir_path
    output_dir_path = set_data_dir_path
    
    if set_data_dir_path is None:
        data_dir_path = get_default_data_dir()  # Only works with expected file structure, otherwise error
    
    if set_output_dir_path is None:
        output_dir_path = make_output_dir(subdir_list=["junction_summaries", "histograms", "cells"]) # creates junction_summaries subdirectory as well

    data_dir_path = data_dir_path.resolve()
    output_dir_path = output_dir_path.resolve()

    return data_dir_path, output_dir_path


def get_directories_in_path(path) -> list[Path]:
    """
    Returns a list of directory names within the specified path.
    """
    directories = []
    try:
        # Get all entries (files and directories) in the path
        entries = os.listdir(path)
        for entry in entries:
            full_path = os.path.join(path, entry)
            # Check if the entry is a directory
            if os.path.isdir(full_path):
                directories.append(full_path)
    except FileNotFoundError:
        print(f"Error: The path '{path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied to access '{path}'.")
    return directories