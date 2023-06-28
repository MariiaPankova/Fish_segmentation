# sorts out images with fish yellow rgb color

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import shutil
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from tqdm.auto import tqdm
from dvc.api import params_show

COLOR_TYPE = tuple[int, int, int]
IMAGE_INPUT_FOLDER = r"data\raw\train_val\images"
MASK_INPUT_FOLDER = r"data\raw\train_val\masks"
IMAGE_OUTNPUT_FOLDER = r"data\clean\images"
MASK_OUTPUT_FOLDER = r"data\clean\masks"

os.makedirs(IMAGE_OUTNPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)


def color_check(image: Image, color: COLOR_TYPE) -> bool:
    colors = {col for _, col in image.getcolors()}
    return tuple(color) in colors


def get_binary_mask(image: Image, color: COLOR_TYPE) -> np.ndarray:
    data = np.asarray(image)
    return np.where(np.equal(data, color).all(axis=2), 1, 0)


def process_image(mask_path: str, color: COLOR_TYPE):
    img = Image.open(mask_path)
    mask_name = osp.basename(mask_path)
    img_name = osp.splitext(osp.basename(mask_path))[0] + ".jpg"
    if color_check(img, color):
        binary_mask = Image.fromarray(get_binary_mask(img, color))
        binary_mask.save(
            osp.splitext(osp.join(MASK_OUTPUT_FOLDER, mask_name))[0] + ".png"
        )
        shutil.copy2(osp.join(IMAGE_INPUT_FOLDER, img_name), IMAGE_OUTNPUT_FOLDER)


def process_folder(mask_color: COLOR_TYPE):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, mask_path, mask_color)
            for mask_path in glob(MASK_INPUT_FOLDER + r"\*.bmp")
        ]
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="processing images"
        ):  # making progressbar
            pass


if __name__ == "__main__":
    params = params_show()
    process_folder(**params["preprocessing"])
