from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import numpy as np
import pytorch_lightning as pl


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
    def __init__(self, backbone_type, head_args, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_type)
        self.backbone.requires_grad = False
        self.head = SegmentationHead(**head_args)


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
        return loss
    
    def validation_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        loss = self.loss_function(prediction.squeeze(), batch["mask"].squeeze())
        self.log("val_loss", loss)

    