from pathlib import Path
import argparse
from tqdm import tqdm
import os
import sys
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(pwd, "../"))

from config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reformat_dataset")
    parser.add_argument("-c", "--config", required=True,
	help="config file")
    args = parser.parse_args()
    config = Config(args.config)
    print(config.__repr__())

    data_dir = Path(config.data_dir)
    label_paths = list(data_dir.glob("*"))
    for label_path in tqdm(label_paths):
        image_paths = list(label_path.glob("*.jpg"))
        for i, image_path in enumerate(image_paths):
            image_path.rename(image_path.parent.joinpath(f"{i}.jpg"))
