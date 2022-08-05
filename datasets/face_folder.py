from pathlib import Path
import random
import os
from tqdm import tqdm
import pickle

from .dataset import Dataset

pwd = os.path.dirname(__file__)
random.seed(10)


# customize from face.evolve
def extract_face_folder(data_dir, cache=None, **kwargs):
    '''
    data_dir
        id1
            *.jpg
        id2
    '''
    data_dir = Path(data_dir)
    data = cache["data"]
    attr_dict = cache["attr_dict"]
    del cache
    
    train = Dataset(data, data_dir, attr_dict, **kwargs)
    val = Dataset([], data_dir, attr_dict, **kwargs)
    test = Dataset([], data_dir, attr_dict, **kwargs)

    return train, val, test, attr_dict


def extract_face_mask_folder(data_dir, data_name, mask_ratio=0.0, cache_path=None, use_cache=True, **kwargs):
    '''
    {data_dir}
        id1
            *.jpg
        id2
    {data_name2}_mask
        id1
        id2
    '''
    data_dir = Path(data_dir)
    if cache_path.is_file() and use_cache:
        print(f"INFO: (face_mask_folder) Load from cache: {str(cache_path)}...")
        cache = torch.load(cache_path)
        data = cache["data"]
        attr_dict = cache["attr_dict"]

    else:
        data = []
        attr_dict = {}
        class_imgs_path = Path(os.path.join(pwd, f"../data/{data_name}_class_imgs.cache"))
        assert class_imgs_path.is_file(), f"{str(class_imgs_path)} not found"
        with open(class_imgs_path, 'rb') as f:
            class_imgs = pickle.load(f)

        label_paths = [data_dir.joinpath(x) for x in class_imgs.keys()]
        i = 0
        for label_path in tqdm(label_paths):
            image_paths = list(label_path.glob("*.jpg"))

            for image_path in image_paths:
                if random.random() <= mask_ratio:
                    path = f"../{data_name}_mask/{image_path.parts[-2]}/{image_path.parts[-1]}"
                else:
                    path = f"../{data_name}/{image_path.parts[-2]}/{image_path.parts[-1]}"
                assert data_dir.joinpath(path).is_file(), f"File {path} not found..."
                data.append([path, i])
            if attr_dict.get(i, None) is None:
                attr_dict[i] = label_path.stem
            i += 1
        random.shuffle(data)
        
        cache = {
            "data": data,
            "attr_dict": attr_dict
        }
        print(f"INFO: (face_mask_folder) Saving cache: {str(cache_path)}...")
        torch.save(cache, cache_path)
    
    train = Dataset(data, data_dir, attr_dict, **kwargs)
    val = Dataset([], data_dir, attr_dict, **kwargs)
    test = Dataset([], data_dir, attr_dict, **kwargs)

    return train, val, test, attr_dict