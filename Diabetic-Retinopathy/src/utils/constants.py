"""Constants related to the repository
"""
from os.path import dirname, join, realpath

# Path to the ultrasound repository. Go up 3 directories: ROOT_DIR/src/utils/constants.py
REPO_PATH = dirname(dirname(dirname(realpath(__file__))))

CONFIG_DIR = join(REPO_PATH, "config")

DATA_ROOT_DIR = "s3://wiai-aiims-private-data/data/"

EYEPACS_ROOT_DIR = join(DATA_ROOT_DIR, "eyepacs")

APTOS_ROOT_DIR = join(DATA_ROOT_DIR, "aptos")

ODIR_ROOT_DIR = join(DATA_ROOT_DIR, "odir")

AIIMS_DELHI_ROOT_DIR = join(DATA_ROOT_DIR, "aiims_delhi")