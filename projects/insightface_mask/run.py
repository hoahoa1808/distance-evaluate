import torch
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import cv2
import argparse
import os
import sys
import h5py
from copy import deepcopy
from pathlib import Path
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(pwd, "../.."))

import settings
from proj_evaluation import eval

from config import Config
from cache import create_cache_data
from pairs import create_pairs
from feature_extraction import FeatureExtraction
from utils.utils import l2_norm

def get_augment(seq):
    def augment(img):
        # image: pil data, rgb
        img = np.array(img).astype(np.uint8)
        img = seq.augment_image(img)
        return Image.fromarray(img)
    return augment
    
def preprocess(img):
    '''
    image: pil data
    return: data preprocessed
    '''
    input_mean = 127.5
    input_std = 127.5
    img = np.array(img).astype(np.uint8)
    # img = img[...,::-1] # BGR to RGB
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = (img - input_mean) / input_std
    
    return img

def postprocess(onnx_output):
    return l2_norm(onnx_output[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="reformat_dataset")
    parser.add_argument("-c", "--config", required=True,
	help="config file")
    args = parser.parse_args()
    config = Config(args.config)
    print(config.__repr__())

    if config.create_cache:
        cache = create_cache_data(config)
    else:
        print(f"INFO: (run) Loading cache: {str(config.cache_path)}...")
        cache = torch.load(config.cache_path)

    if config.create_pairs:
        create_pairs(config, cache)

    # create emb for data_prefix
    if config.extract_feature:
        config_bak = deepcopy(config)
        for prefix in settings.data_prefix:
            config.data_name = config_bak.data_name + prefix
            config.data_dir = Path(str(config_bak.data_dir) + prefix)
            config.embs_h5 = Path(config.cache_dir).joinpath(f"{config.data_name}_{config.model_name}_{config.prefix}.h5")
            
            if config.embs_h5.is_file():
                print(f"INFO: (run) embs_h5 is existed, ignoring{str(config.embs_h5)}")
                continue
    
            print(f"INFO: (run) data_dir: {str(config.data_dir)}...")
            seq = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontally flip 50% of the images
                iaa.Sometimes(0.8, iaa.MotionBlur((5, 15))),
                iaa.Sometimes(0.3, iaa.AdditiveLaplaceNoise())
            ])
            augment = get_augment(seq)
            fe = FeatureExtraction(config, cache=cache, preprocess=preprocess, augment=augment, postprocess=postprocess)
            fe.run()
        config = deepcopy(config_bak)

    embs_dict = {}
    idx_dict = {}
    config_bak = deepcopy(config)
    for pair in settings.eval_pairs:
        for prefix in pair.values():
            if embs_dict.get(prefix, None) is None:
                config.data_name = config_bak.data_name + prefix
                config.embs_h5 = Path(config.cache_dir).joinpath(f"{config.data_name}_{config.model_name}_{config.prefix}.h5")

                print(f"INFO: (run) Loading vector db {str(config.embs_h5)}...")
                db = h5py.File(config.embs_h5)
                embs_dict[prefix] = db['embs'][:]
                idx = db['ids'][:]
                idx_dict[prefix] = dict(zip(idx, list(range(idx.shape[0]))))
                db.close()

        prefix1, prefix2 = pair["item1"], pair["item2"]
        config.result = Path(config.cache_dir).joinpath(f"{config.data_name}_{config.model_name}_{config.prefix}_{prefix1}_{prefix2}.jpg")
        eval(
            config, cache["total_pairs"],
            embs_dict[prefix1], idx_dict[prefix1],
            embs_dict[prefix2], idx_dict[prefix2]
        )
    config = deepcopy(config_bak)
    

    # eval(config, cache["total_pairs"])
