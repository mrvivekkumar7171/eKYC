import logging, os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


with open(PROJECT_ROOT / "config.yaml") as yaml_file:
    config = yaml.safe_load(yaml_file)

artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = PROJECT_ROOT / artifacts['LOG_DIR']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=log_dir / log_file_name, level=logging.INFO, format=logging_str, filemode="a")


def file_exists(file_path):
    is_exist = os.path.exists(file_path)
    if is_exist:
        logging.info(f"File is exist for {file_path}")
        return True
    logging.info(f"File doesn't exist for {file_path}")
    return False

def read_yaml(path_to_yaml:str) -> dict:
    yaml_path = Path(path_to_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path
    with open(yaml_path) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {yaml_path} loaded successfully")
    return content

def artifact_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def create_dirs(dirs: list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        logging.info(f"Directory is created at {dir}")