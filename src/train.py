from core.dataset import FishDataset, get_mixed_dataset, get_test_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import albumentations as A
from dvc.api import params_show
import torch
from core.model import LITFishSegmentation
from pytorch_lightning import Trainer
from dvclive import Live
from dvclive.lightning import DVCLiveLogger
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == "__main__":
    pl.seed_everything(42)
    params = params_show()
    transforms = A.from_dict(params["transform"])
    train_data, val_data = get_mixed_dataset(**params["dataset"], transforms=transforms)

    test_data = get_test_dataset(**params["test_dataset"])
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)
    train_dataloader = DataLoader(
        train_data, batch_size=16, shuffle=True, num_workers=2, persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_data, batch_size=16, shuffle=False, num_workers=1, persistent_workers=True
    )

    model = LITFishSegmentation(**params["model"])
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True)

    with Live() as live:
        os.makedirs("dvclive\plots\custom", exist_ok=True)
        trainer = Trainer(
            accelerator="gpu",
            logger=DVCLiveLogger(experiment=live),
            default_root_dir="weights",
            callbacks=[checkpoint_callback],
            max_epochs=500,
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        trainer.test(
            model=model,
            dataloaders=test_dataloader,
            ckpt_path="best",
        )
