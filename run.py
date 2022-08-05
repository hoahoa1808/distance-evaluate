import torch
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import cv2
import argparse

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
    input_mean = np.asarray([0.485, 0.456, 0.406])
    input_std = np.asarray([0.229, 0.224, 0.225])
    img = np.array(img).astype(np.uint8)
    img = cv2.resize(img, (128, 256))
    img = img.astype(np.float32) / 255.
    img = (img - input_mean) / input_std
    # img = img[...,::-1] # BGR to RGB
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    
    return img

def postprocess(onnx_output):
    onnx_output = np.squeeze(onnx_output[0])  # ['batch_size', 768, 1, 1] -> ['batch_size', 768]
    return l2_norm(onnx_output)

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
            # iaa.Sometimes(0.8, iaa.MotionBlur((5, 15))),
            iaa.Sometimes(0.3, iaa.AdditiveLaplaceNoise())
        ])
        augment = get_augment(seq)
        fe = FeatureExtraction(config, cache=cache, preprocess=preprocess, augment=augment, postprocess=postprocess)
        fe.run()
    
    eval(config, cache["total_pairs"])
