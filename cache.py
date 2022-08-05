import torch
from pathlib import Path
from tqdm import tqdm

from config import Config


def create_cache_data(config):
    cache_path = Path(config.cache_path)
    data_dir = Path(config.data_dir)
    min_imgs = config.min_imgs
    data = []
    attr_dict = {}
    class_imgs = {}
    
    print("INFO: (cache) Creating cache...")
    class_idx = 0
    label_paths = list(data_dir.glob("*"))
    for label_path in tqdm(label_paths):
        image_paths = list(label_path.glob("*.jpg"))
        if label_path.is_file() or len(image_paths) < min_imgs:
            continue
        class_imgs[label_path.stem] = len(image_paths)
    
        for image_path in image_paths:
            data.append([f"{image_path.parts[-2]}/{image_path.parts[-1]}", class_idx])
        if attr_dict.get(class_idx, None) is None:
            attr_dict[class_idx] = label_path.stem
        class_idx += 1
    
    cache = {
        "data": data,   # list of [(image_path, cls_id),...]
        "attr_dict": attr_dict,
        "class_imgs": class_imgs   # dict of {"cls_name": num_images}
    }
    print(f"INFO: (cache) Saving cache: {str(cache_path)}...")
    torch.save(cache, cache_path)

    return cache


if __name__ == '__main__':
    config_path = "/research/classification/distance_evaluation/config.yml"
    config = Config(config_path)

    create_cache_data(config)
