import argparse
import os

from omegaconf import OmegaConf

def get_file_list(directory, preserve_ext=False):
    """
    Get a list of filenames in a directory.

    Args:
        directory (str): Path to the directory.
        preserve_ext (bool, optional): Whether to preserve file extensions in the output. Defaults to False.

    Returns:
        list: A list of filenames (with or without extensions).
    """
    file_list = []

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            if preserve_ext:
                file_list.append(filename)
            else:
                name, _ = os.path.splitext(filename)
                file_list.append(name)

    return file_list

parser = argparse.ArgumentParser(description="Print a list of filenames in a directory as a YAML map.")

parser.add_argument("--dir", type=str, default=".", help="Path to the directory")
parser.add_argument("--keep_extension", action="store_true", help="Preserve file extensions in the output")

args = parser.parse_args()

file_list = get_file_list(args.dir, args.keep_extension)

# Create a dictionary with the key "data" and the list of filenames as the value
data_map = {"data": file_list}
yaml_output = OmegaConf.create(data_map)

print(OmegaConf.to_yaml(yaml_output))
