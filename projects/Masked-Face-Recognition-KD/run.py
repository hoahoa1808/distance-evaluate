import torch
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import cv2
import argparse
import os
import sys
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(pwd, "../.."))

from config import Config
from cache import create_cache_data
from pairs import create_pairs
from feature_extraction import FeatureExtraction
from evaluation import eval
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
    
    if config.extract_feature:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Sometimes(0.8, iaa.MotionBlur((5, 15))),
            iaa.Sometimes(0.3, iaa.AdditiveLaplaceNoise())
        ])
        augment = get_augment(seq)
        fe = FeatureExtraction(config, cache=cache, preprocess=preprocess, augment=augment, postprocess=postprocess)
        fe.run()
    
    eval(config, cache["total_pairs"])
