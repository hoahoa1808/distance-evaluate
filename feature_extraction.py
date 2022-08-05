import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import torch
import timeit
import psutil
import cv2

from config import Config
from datasets import init_dataset
from utils.onnx_inference import OnnxInfer, HDF5DatasetWriter
pwd = os.path.dirname(os.path.realpath(__file__))


class FeatureExtraction:
    def __init__(self, config, cache, preprocess, postprocess=None, augment=None, batch_size=128):
        dataset = init_dataset(
            name=config.data_type, data_dir=config.data_dir,
            cache=cache
        )[0]
        dataset.augmentation = augment
        dataset.transform = preprocess
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, pin_memory=True,
            num_workers=psutil.cpu_count(), drop_last=False
        )

        self.ems_writer = HDF5DatasetWriter(length=len(dataset), outputPath=config.embs_h5, embDim=config.emb_dim)
        
        self.model = OnnxInfer(weight_paths=config.onnx_path)
        self.postprocess = postprocess

        print(f"INFO: Extracting features {str(config.embs_h5)}...")
    
    
    def run(self):
        for i, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            t0 = timeit.default_timer()
            embs = self.model(inputs.numpy())
            if self.postprocess is not None:
                embs = self.postprocess(embs)
            t1 = timeit.default_timer()

            cls_ids, cls, img_ids = labels
            cls_ids = cls_ids.numpy()
            img_ids = np.array(list(map(int, img_ids)))
            ids = (cls_ids+1e6)*1e4+img_ids
            ids = ids.astype(np.int64).tolist()
            self.ems_writer.add(embs, ids)

        self.ems_writer.close()


if __name__ == '__main__':
    import yaml
    from imgaug import augmenters as iaa
    from PIL import Image

    from utils.utils import l2_norm

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        # iaa.Sometimes(0.8, iaa.MotionBlur((5, 15))),
        iaa.Sometimes(0.3, iaa.AdditiveLaplaceNoise())
    ])
    
    def postprocess(onnx_output):
        return l2_norm(onnx_output[0])
    
    config_path = "/research/classification/distance_evaluation/config.yml"
    config = Config(config_path)

    cache = torch.load(config.cache_path)
    fe = FeatureExtraction(config, cache=cache, preprocess=preprocess, augment=augment, postprocess=postprocess)
    fe.run()
    
    
    
        
