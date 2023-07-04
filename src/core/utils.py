from dvc.api import params_show
import albumentations as A
from core.dataset import get_mixed_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.utils import draw_segmentation_masks, save_image
import torchvision.transforms.functional as F
import torch


def draw_batch(batch: dict[str, torch.Tensor], filename: str):
    result = []
    for image, mask in zip(
        F.convert_image_dtype(
            F.normalize(
                batch["image"].to("cpu"),
                mean=[
                    -0.485,
                    -0.456,
                    -0.406,
                ],
                std=[
                    1 / 0.229,
                    1 / 0.224,
                    1 / 0.225,
                ],
            ),
            torch.uint8,
        ),
        batch["mask"].to(bool).to("cpu"),
    ):
        result.append(draw_segmentation_masks(image, mask, colors=["yellow"]))
    save_image(F.convert_image_dtype(torch.stack(result), torch.float), filename)


if __name__ == "__main__":
    pl.seed_everything(42)
    params = params_show()
    transforms = A.from_dict(params["transform"])
    train_data, val_data = get_mixed_dataset(**params["dataset"], transforms=transforms)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    batch = next(iter(train_dataloader))
    draw_batch(batch, "test.jpg")
