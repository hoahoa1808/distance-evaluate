from pathlib import Path
import os
import torch
import numpy as np
from tqdm import tqdm

from config import Config


def _create_positive(n_positive, class_imgs):
    ids = np.random.randint(0, len(class_imgs), size=(n_positive))
    len_face_imgs = np.array(list(class_imgs.values()))[ids]
    max_id = max(list(class_imgs.values()))

    img_ids = np.random.randint(0, max_id, (n_positive, 2)) % len_face_imgs[..., np.newaxis]
    ids = np.stack([ids, ids], 1)
    ids = (ids+1e6)*1e4+img_ids
    ids = ids.astype(np.int64)
    
    # filt existed pair
    ids = ids[ids[:, 0]!=ids[:, 1]]
    ids = np.sort(ids, axis=1)
    ids = np.unique(ids, axis=0)
    
    data = np.concatenate([ids, np.ones((ids.shape[0], 1), ids.dtype)], 1)

    return data


def _create_negative(n_negative, class_imgs):
    ids = np.random.randint(0, len(class_imgs), size=(n_negative, 2))
    ids = ids[ids[:, 0] != ids[:, 1]]
    n_negative = ids.shape[0]
    len_face_imgs = np.array(list(class_imgs.values()))[ids]
    max_id = max(list(class_imgs.values()))

    img_ids = np.random.randint(0, max_id, (n_negative, 2)) % len_face_imgs
    ids = (ids+1e6)*1e4+img_ids
    ids = ids.astype(np.int64)
    ids = np.sort(ids, axis=1)
    ids = np.unique(ids, axis=0)
    data = np.concatenate([ids, np.zeros((ids.shape[0], 1), ids.dtype)], 1)

    return data


def create_pairs(config, cache):
    """_summary_
    """
    pairs_csv = Path(config.pairs_csv)
    total = int(config.total)
    positive_ratio = config.positive_ratio
    buffer_size = int(config.buffer_size)
    if pairs_csv.is_file():
        pairs_csv.unlink()
        print(f"INFO: (pairs) Remove old pairs_csv data: {str(pairs_csv)}...")
    class_imgs = cache["class_imgs"]

    print(f"INFO: (pairs) Creating pairs: {str(pairs_csv)}...")
    total_pairs = 0
    for i in tqdm(range(0, total, buffer_size)):
        n_pairs = total - i
        n_pairs = min(n_pairs, buffer_size)
        n_positive = int(positive_ratio*n_pairs)
        n_negative = n_pairs-n_positive

        positive_pairs = _create_positive(n_positive, class_imgs=class_imgs)
        negative_pairs = _create_negative(n_negative, class_imgs=class_imgs)
        data = np.concatenate([negative_pairs, positive_pairs], axis=0)
        total_pairs += data.shape[0]
        if pairs_csv.is_file():
            with open(pairs_csv, "ab") as f:
                np.savetxt(f, data, delimiter=',', comments="", fmt='%i')
        else:
            np.savetxt(pairs_csv, data, delimiter=',', header="item1,item2,label", comments="", fmt='%i')
    
    print(f"INFO: (pairs) Updating cache data total_pairs={total_pairs}...")
    cache["total_pairs"] = total_pairs
    torch.save(cache, config.cache_path)

    return total_pairs


if __name__ == '__main__':
    config_path = "/research/classification/distance_evaluation/config.yml"
    config = Config(config_path)

    cache = torch.load(config.cache_path)
    total_pairs = create_pairs(config, cache)
    print("Done...")
