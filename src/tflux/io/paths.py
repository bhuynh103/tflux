import sys
import os

def get_directories_in_path(path):
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


### DIRECTORY PATHS ###
# "data/raw/WT"
# "data/raw/WT-selected"
# "data/raw/WTvsBleb_experimental"
# "data/raw/WTvsBleb_control"
# "data/raw/WTvsLabB_0.5uMLatB_experimental"
# "data/raw/WTvsLabB_0.5uMLatB_control"