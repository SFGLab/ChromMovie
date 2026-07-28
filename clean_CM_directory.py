import os
import shutil
import yaml
import numpy as np
import argparse


def clean_directory(yaml_path):
    """
    Deletes all the files in the output directory from the ChromMovie simulation except for the cif files of the structures from the last step.
    THe output directory is determined from the YAML file in yaml_path.
    All of the files in the determined output directory are expected to contain a specific ChromMovie simulation output.
    """

    # Read the output path to be cleaned from yaml file. 
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    target_dir = config_data['general']['output']

    # Safety Check: Convert to absolute path and verify existence
    target_dir = os.path.abspath(target_dir)
    if not os.path.exists(target_dir):
        print(f"Error: The directory '{target_dir}' does not exist.")
        return

    # Iterate through everything inside the target directory
    for item in os.listdir(target_dir):
        if not item.endswith("yaml"):
            item_path = os.path.join(target_dir, item)

            # Check if this is the frames_cif folder and delete all files except last step
            if item == "frames_cif":
                frames = os.listdir(item_path)
                max_step = np.max([int(x.split("_")[0][4:]) for x in frames])
                for frame in frames:
                    if not frame.startswith(f"step{str(max_step).zfill(3)}"):
                        os.remove(os.path.join(item_path, frame))
                continue

            try:
                # If it's a directory (and not our protected one), delete it and its contents
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                # If it's a file or a symlink, delete it
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"[ERROR] Could not delete {item}: {e}")
    # print(f"Folder {target_dir} cleared successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deletes all the files in target_dir from the ChromMovie simulation except for necessary ones."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the yaml file.",
    )

    args = parser.parse_args()

    clean_directory(args.input)
