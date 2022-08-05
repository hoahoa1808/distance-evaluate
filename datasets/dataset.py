from PIL import Image
from pathlib import Path

class Dataset(object):

    def __init__(
            self,
            imgs,
            data_dir,
            attr_dict,
            augmentation=None,
            transform=None,
            **kwargs
        ):
        self.data_dir = Path(data_dir)
        self.classes = attr_dict
        self.augmentation = augmentation
        self.transform = transform

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, cls_id = self.imgs[index]
        img_path = self.data_dir.joinpath(img_path)
        img = Image.open(str(img_path)).convert('RGB')
        attrs = [cls_id, self.classes[cls_id], img_path.stem]

        if self.augmentation is not None:
            img = self.augmentation(img)
        # output = Path(str(img_path).replace(img_path.parts[-3], img_path.parts[-3]+"_aug"))
        # output.parent.mkdir(parents=True, exist_ok=True)
        # img.save(output)
        if self.transform is not None:
            img = self.transform(img)

        return img.copy(), attrs
