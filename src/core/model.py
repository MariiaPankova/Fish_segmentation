from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from core.utils import draw_batch
import pytorch_optimizer
import segmentation_models_pytorch as smp


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding="same"
        )
        self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, padding="same"
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class FishUnet(torch.nn.Module):
    def __init__(self, in_channels=3, output_channels=2, kernel_size=(3, 3)):
        super().__init__()
        # 224x224
        self.conv_down1 = ConvBlock(in_channels, 64, kernel_size)
        self.max_pool1 = torch.nn.MaxPool2d((2, 2), stride=2)  # 112x112

        self.conv_down2 = ConvBlock(64, 128, kernel_size)
        self.max_pool2 = torch.nn.MaxPool2d((2, 2), stride=2)  # 56x56

        self.conv_down3 = ConvBlock(128, 256, kernel_size)
        self.max_pool3 = torch.nn.MaxPool2d((2, 2), stride=2)  # 28x28

        self.conv_down4 = ConvBlock(256, 512, kernel_size)

        self.conv_transpose3 = torch.nn.ConvTranspose2d(
            512, 256, kernel_size=(2, 2), stride=2
        )  # 56x56
        self.conv_up3 = ConvBlock(512, 256, kernel_size)

        self.conv_transpose2 = torch.nn.ConvTranspose2d(
            256, 128, kernel_size=(2, 2), stride=2
        )  # 112x112
        self.conv_up2 = ConvBlock(256, 128, kernel_size)

        self.conv_transpose1 = torch.nn.ConvTranspose2d(
            128, 64, kernel_size=(2, 2), stride=2
        )  # 224x224
        self.conv_up1 = ConvBlock(128, 64, kernel_size)

        self.final_conv = torch.nn.Conv2d(64, output_channels, kernel_size=(1, 1))

        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        x_down1 = self.conv_down1(x)
        x_down2 = self.max_pool1(x_down1)

        x_down2 = self.conv_down2(x_down2)
        x_down3 = self.max_pool2(x_down2)

        x_down3 = self.conv_down3(x_down3)
        x_down4 = self.max_pool3(x_down3)

        x_down4 = self.conv_down4(x_down4)

        x_up3 = self.conv_transpose3(x_down4)
        # print(x_up3.shape, x_down3.shape)
        x_up3 = torch.concatenate([x_up3, x_down3], dim=1)
        x_up3 = self.conv_up3(x_up3)

        x_up2 = self.conv_transpose2(x_up3)
        x_up2 = torch.concatenate([x_up2, x_down2], dim=1)
        x_up2 = self.conv_up2(x_up2)

        x_up1 = self.conv_transpose1(x_up2)
        x_up1 = torch.concatenate([x_up1, x_down1], dim=1)
        x_up1 = self.conv_up1(x_up1)

        res = self.final_conv(x_up1)
        res = self.softmax(res)
        return res


class LITFishSegmentation(pl.LightningModule):
    def __init__(self, model_args, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.model = FishUnet(**model_args)
        self.val_batch: torch.Tensor | None = None
        self.train_iou = torchmetrics.JaccardIndex("binary")
        self.val_iou = torchmetrics.JaccardIndex("binary")
        self.test_iou = torchmetrics.JaccardIndex("binary")
        self.save_hyperparameters()

    def forward(self, batch) -> Any:
        pred_mask = self.model(batch["image"])
        return pred_mask

    def configure_optimizers(self) -> Any:
        optimizer = pytorch_optimizer.Ranger(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        loss = self.loss_function(prediction, batch["mask"])
        self.log("train_loss", loss)
        self.train_iou(prediction.argmax(dim=1), batch["mask"].argmax(dim=1))
        return loss

    def on_train_epoch_end(self):
        self.log("train_iou", self.train_iou)

    def validation_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        loss = self.loss_function(prediction, batch["mask"])
        self.val_iou(prediction.argmax(dim=1), batch["mask"].argmax(dim=1))
        self.log("val_loss", loss)
        if batch_idx == 0:
            self.val_batch = {"image": batch["image"], "mask": prediction}

    def on_validation_epoch_end(self):
        draw_batch(
            self.val_batch, f"dvclive\plots\custom\epoch_{self.current_epoch}.jpg"
        )
        self.log("val_iou", self.val_iou)

    def test_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        loss = self.loss_function(prediction, batch["mask"])
        self.test_iou(prediction.argmax(dim=1), batch["mask"].argmax(dim=1))
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        self.log("test_iou", self.test_iou)
