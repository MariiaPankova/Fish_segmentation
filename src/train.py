from core.dataset import FishDataset, get_mixed_dataset, get_transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import albumentations as A
from dvc.api import params_show
import torch
from core.model import LITFishSegmentation
from pytorch_lightning import Trainer
from dvclive import Live
from dvclive.lightning import DVCLiveLogger


if __name__ == "__main__":
    params = params_show()
    transforms = A.from_dict(params["transform"])
    train_data, val_data = get_mixed_dataset(**params["dataset"], transforms=transforms
    )
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = LITFishSegmentation(**params["model"])
    with Live() as live:
        trainer = Trainer(accelerator="gpu", logger=DVCLiveLogger(experiment=live), default_root_dir="weights")
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)