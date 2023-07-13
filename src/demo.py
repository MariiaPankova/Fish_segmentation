import gradio as gr
from core.model import LITFishSegmentation
from core.dataset import get_transforms
import albumentations as A
from core.utils import invTrans
import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks


model = LITFishSegmentation.load_from_checkpoint(
    r"weights\DvcLiveLogger\dvclive_run\checkpoints\epoch=436-step=23598.ckpt"
)
model.to("cuda")
transforms = get_transforms()


def segment(image):
    image_transformed = transforms(image=image)["image"].unsqueeze(0)
    image_transformed = image_transformed.to("cuda")
    prediction = model({"image": image_transformed})
    background = F.convert_image_dtype(
        invTrans(
            image_transformed.to("cpu"),
        ),
        torch.uint8,
    )
    mask = prediction.argmax(dim=1).to(bool).to("cpu")
    output = draw_segmentation_masks(background[0], mask[0], colors=["yellow"])
    output = output.permute((1, 2, 0)).numpy()
    return output


gr.Interface(fn=segment, inputs="image", outputs="image").launch()
