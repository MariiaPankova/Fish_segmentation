from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from core.utils import draw_batch


class SegmentationHead(torch.nn.Module):
    def __init__(self, input_shape: int = 384, output_shape=(224, 224), scale_factor=7):
        super(SegmentationHead, self).__init__()
        self.inner_shape = (
            output_shape[0] // scale_factor,
            output_shape[1] // scale_factor,
        )
        print(np.prod(self.inner_shape))
        self.linear1 = torch.nn.Linear(input_shape, np.prod(self.inner_shape))
        self.sigmoid = torch.nn.Sigmoid()
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor)

    def forward(self, x):
        x = self.linear1(x)
        x = x.reshape((len(x), 1, *self.inner_shape))
        x = self.sigmoid(x)
        x = self.upsample(x)
        return x


class LITFishSegmentation(pl.LightningModule):
    def __init__(self, backbone_type, head_args, learning_rate, threshold) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_type)
        self.backbone.requires_grad = False
        self.head = SegmentationHead(**head_args)
        self.val_batch: torch.Tensor | None = None
        self.threshold = threshold
        self.iou = torchmetrics.JaccardIndex("binary", threshold=self.threshold)

    def forward(self, batch) -> Any:
        self.backbone.eval()
        with torch.no_grad():
            embedding = self.backbone(batch["image"])
        pred_mask = self.head(embedding)
        return pred_mask

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        loss = self.loss_function(prediction.squeeze(), batch["mask"].squeeze())
        self.log("train_loss", loss)
        self.iou(prediction.squeeze(), batch["mask"].squeeze())
        self.log("train_step_iou", self.iou)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        loss = self.loss_function(prediction.squeeze(), batch["mask"].squeeze())
        self.iou(prediction.squeeze(), batch["mask"].squeeze())
        self.log("val_loss", loss)
        pred_mask = torch.where(prediction > self.threshold, 1, 0)
        self.val_batch = {"image": batch["image"], "mask": pred_mask}

    def on_validation_epoch_end(self):
        draw_batch(
            self.val_batch, f"dvclive\plots\custom\epoch_{self.current_epoch}.jpg"
        )
        self.log("val_iou", self.iou)
