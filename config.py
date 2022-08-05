import yaml
from pathlib import Path
from tabulate import tabulate
import pprint
import os
pwd = os.path.dirname(os.path.realpath(__file__))


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # common fields
        self.data_name = config["data_name"]
        self.cache_dir = config.get("cache_dir", os.path.join(pwd, "data"))
        self.cache_dir = Path(self.cache_dir).joinpath(self.data_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir.joinpath(f"{self.data_name}_data.cache")
        self.data_dir = Path(config["data_dir"])
        self.prefix = config["prefix"]

        # cache
        self.create_cache = config["create_cache"]
        self.min_imgs = config["min_imgs"]

        # pairs
        self.create_pairs = config["create_pairs"]
        self.total = int(config["total"])
        self.positive_ratio = config["positive_ratio"]
        self.pairs_csv = self.cache_dir.joinpath(f"{self.data_name}.csv")  # output pairs file
        self.buffer_size = int(config["buffer_size"])

        # feature_exaction
        self.extract_feature = config["extract_feature"]
        self.data_type = config["data_type"]  # type func for dataloader
        self.onnx_path = config["onnx_path"]
        self.emb_dim = config["emb_dim"]
        self.model_name = config["model_name"]
        self.embs_h5 = Path(self.cache_dir).joinpath(f"{self.data_name}_{self.model_name}_{self.prefix}.h5")

        # eval
        self.result = Path(self.cache_dir).joinpath(f"{self.data_name}_{self.model_name}_{self.prefix}.jpg")
    

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")
