import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from glob import glob

from natsort import natsorted
import yaml


def create_experiment_name(prefix=None, suffix=None):    
    experiment_name = ""

    if prefix:
        if isinstance(prefix, list):
            prefix = "_".join(prefix)
        
        experiment_name += prefix + "_"

    experiment_name += datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    if suffix:
        if isinstance(suffix, list):
            suffix = "_".join(suffix)

        experiment_name += "_" + suffix

    return experiment_name


def copy_configurations_to_dir(cfg, directory, write_yaml=True, write_datalist=True):
    if write_yaml:
        with open(os.path.join(directory, "config.yaml"), 'w+') as yaml_file:
            yaml.dump(cfg, yaml_file, default_flow_style=False)

    if write_datalist:
        src_path = os.path.join(cfg.datalists_base_path, cfg.data_list)
        dst_path = os.path.join(directory, "datalist")
        os.makedirs(dst_path, exist_ok=True)

        files = glob(os.path.join(src_path,"*.txt"))
        for f in files:
            list_name = os.path.basename(f)
            copyfile(f, os.path.join(dst_path, list_name))