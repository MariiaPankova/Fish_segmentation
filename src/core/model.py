from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from core.utils import draw_batch


class SegmentationHead(torch.nn.Module):
    def __init__(
        self,
        input_shape: int = 384,
        output_shape=(224, 224),
        inner_shape=1024,
        scale_factor=7,
    ):
        super(SegmentationHead, self).__init__()
        self.output_shape = output_shape
        self.scale_factor = scale_factor
        self.linear1 = torch.nn.Linear(input_shape, inner_shape)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(
            inner_shape, np.prod(output_shape) // scale_factor**2 * 2
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear")

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.reshape(
            (
                len(x),
                2,
                self.output_shape[0] // self.scale_factor,
                self.output_shape[1] // self.scale_factor,       
            )
        )
        x = self.softmax(x)
        x = self.upsample(x)
        return x


class LITFishSegmentation(pl.LightningModule):
    def __init__(self, backbone_type, head_args, learning_rate, threshold) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.BCELoss()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_type)
        self.head = SegmentationHead(**head_args)
        self.val_batch: torch.Tensor | None = None
        self.threshold = threshold
        self.train_iou = torchmetrics.JaccardIndex("binary", threshold=self.threshold)
        self.val_iou = torchmetrics.JaccardIndex("binary", threshold=self.threshold)
        self.test_iou = torchmetrics.JaccardIndex("binary", threshold=self.threshold)
        self.backbone.requires_grad_(False)
        self.save_hyperparameters()

    def forward(self, batch) -> Any:
        self.backbone.eval()
        with torch.no_grad():
            embedding = self.backbone(batch["image"])
        pred_mask = self.head(embedding)
        return pred_mask

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.(self.parameters(), lr=self.learning_rate)
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
        pred_mask = torch.where(prediction > self.threshold, 1, 0)

        if batch_idx == 0:
            self.val_batch = {"image": batch["image"], "mask": pred_mask}

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
