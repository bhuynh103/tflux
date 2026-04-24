import os
from datetime import datetime
from pathlib import Path
from tflux.utils.logging import get_logger

logger = get_logger(__name__)


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


def make_output_dir(sample_labels: list[str]) -> Path:
    """
    Creates a unique run directory with sample-specific subdirectories.
    
    Structure:
    outputs/YYYY-MM-DD_NNN/
    ├── comparisons/
    ├── control/
    │   ├── histograms/
    │   └── cells/
    └── experimental/ ...
    """
    root_path: Path = find_root()
    outputs_dir = root_path / "outputs" 
    outputs_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')
    run_number = 1

    while True:
        dir_name = f"{date_str}_{run_number:03d}"
        run_dir = outputs_dir / dir_name
        
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Create the global comparisons folder
            (run_dir / "comparisons").mkdir(exist_ok=True)
            
            # 2. Create nested folders for each sample label
            for label in sample_labels:
                sample_folder = run_dir / label
                sample_folder.mkdir()
                
                # Standard sub-folders within each sample
                (sample_folder / "histograms").mkdir()
                (sample_folder / "cells").mkdir()
                
            return run_dir
        
        run_number += 1


def prepare_io(
    data_paths: dict[str, Path], 
    set_output_dir_path: Path = None
) -> tuple[dict[str, Path], Path]:
    """
    Prepares input/output paths for multiple samples.
    
    Parameters
    ----------
    data_paths : dict
        A mapping of labels to paths, e.g., 
        {"WT": Path("..."), "control": Path("..."), "experimental": Path("...")}
    set_output_dir_path : Path, optional
        The root directory for all outputs.

    Returns
    -------
    resolved_inputs : dict
        Dictionary with resolved absolute Paths.
    output_root : Path
        Resolved absolute Path to the output directory.
    """
    
    # 1. Resolve Output Directory
    if set_output_dir_path is None:
        output_root = make_output_dir(sample_labels=list(data_paths.keys()))
    else:
        output_root = set_output_dir_path
    
    output_root = output_root.resolve()

    # 2. Resolve Input Directories
    resolved_inputs = {}
    for label, path in data_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Input path for {label} does not exist: {path}")
        resolved_inputs[label] = path.resolve()

    return resolved_inputs, output_root


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
        logger.error(f"The path '{path}' was not found.")
    except PermissionError:
        logger.error(f"Permission denied to access '{path}'.")
    return directories