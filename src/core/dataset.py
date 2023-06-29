from torch.utils.data import Dataset, Subset, ConcatDataset, random_split
from glob import glob
import settings
from PIL import Image
import os.path as osp
from typing import Literal
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FishDataset(Dataset):
    def __init__(
        self, image_type: Literal["empty", "default"] = "default", transform=None
    ) -> None:
        self.image_type = image_type
        self.transform = transform
        if self.image_type == "default":
            self.image_paths = glob(settings.IMAGE_OUTPUT_FOLDER + r"\*.jpg")
        elif self.image_type == "empty":
            self.image_paths = glob(settings.EMPTY_IMAGE_OUTPUT_FOLDER + r"\*.jpg")
        else:
            raise ValueError("Unknown image type")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> dict:
        image_path = self.image_paths[index]
        image = np.array(Image.open(image_path))
        if self.image_type == "empty":
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            mask_path = osp.join(
                settings.MASK_OUTPUT_FOLDER,
                osp.splitext(osp.basename(image_path))[0] + ".png",
            )
            mask = np.array(Image.open(mask_path), dtype=np.float32)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return {"image": image, "mask": mask}


def get_mixed_dataset(
    empty_images_ratio: float,
    train_val_ratio: float,
    transforms: A.Compose | None = None,
):
    default_dataset = FishDataset(transform=transforms)
    empty_images_dataset = FishDataset(image_type="empty", transform=transforms)
    empty_images_count = int(len(default_dataset) * empty_images_ratio)
    indices = np.random.choice(
        range(len(empty_images_dataset)), empty_images_count, replace=False
    )
    empty_subset = Subset(empty_images_dataset, indices)
    default_X, default_Y = random_split(
        default_dataset, [train_val_ratio, 1 - train_val_ratio]
    )
    empty_X, empty_Y = random_split(
        empty_subset, [train_val_ratio, 1 - train_val_ratio]
    )
    return ConcatDataset([default_X, empty_X]), ConcatDataset([default_Y, empty_Y])


def get_transforms():
    compose = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return compose
