from torch.utils.data import Dataset, Subset, ConcatDataset, random_split
from glob import glob
import settings
from PIL import Image
import os.path as osp
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torch


class FishDataset(Dataset):
    def __init__(
        self, image_dir: str, mask_dir: str | None = None, transform=None
    ) -> None:
        self.transform = transform
        self.image_paths = glob(image_dir + r"\*.jpg")
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> dict:
        image_path = self.image_paths[index]
        image = np.array(Image.open(image_path))
        if self.mask_dir is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            mask_path = osp.join(
                self.mask_dir,
                osp.splitext(osp.basename(image_path))[0] + ".png",
            )
            mask = np.array(Image.open(mask_path), dtype=np.float32)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        mask = F.one_hot(mask.to(torch.long), 2).to(torch.float).permute((2, 0, 1))
        return {"image": image, "mask": mask}


def get_mixed_dataset(
    empty_images_ratio: float,
    train_val_ratio: float,
    transforms: A.Compose | None = None,
):
    default_dataset = FishDataset(
        image_dir=settings.IMAGE_OUTPUT_FOLDER,
        mask_dir=settings.MASK_OUTPUT_FOLDER,
        transform=transforms,
    )
    empty_images_dataset = FishDataset(
        image_dir=settings.EMPTY_IMAGE_OUTPUT_FOLDER, transform=transforms
    )
    empty_images_count = int(len(default_dataset) * empty_images_ratio)
    indices = np.random.choice(
        range(len(empty_images_dataset)), empty_images_count, replace=False
    )
    empty_subset = Subset(empty_images_dataset, indices)
    default_train, default_val = random_split(
        default_dataset, [train_val_ratio, 1 - train_val_ratio]
    )
    empty_train, empty_val = random_split(
        empty_subset, [train_val_ratio, 1 - train_val_ratio]
    )
    return ConcatDataset([default_train, empty_train]), ConcatDataset(
        [default_val, empty_val]
    )


def get_test_dataset(empty_images_ratio: float):
    compose = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    test_dataset = FishDataset(
        image_dir=settings.TEST_IMAGE_OUTPUT_FOLDER,
        mask_dir=settings.TEST_MASK_OUTPUT_FOLDER,
        transform=compose,
    )
    empty_test_dataset = FishDataset(
        image_dir=settings.TEST_EMPTY_IMAGE_OUTPUT_FOLDER, transform=compose
    )

    empty_images_count = int(len(test_dataset) * empty_images_ratio)
    indices = np.random.choice(
        range(len(empty_test_dataset)), empty_images_count, replace=False
    )
    empty_subset = Subset(empty_test_dataset, indices)

    return ConcatDataset([test_dataset, empty_subset])


def get_transforms():
    compose = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return compose
