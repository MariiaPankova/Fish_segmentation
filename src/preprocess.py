# sorts out images with fish yellow rgb color

from PIL import Image
import numpy as np
import os.path as osp
import shutil
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from tqdm.auto import tqdm
from dvc.api import params_show
import settings

COLOR_TYPE = tuple[int, int, int]


class DataPreprocess:
    def __init__(
        self,
        image_input: str,
        mask_input: str,
        image_output: str,
        mask_output: str,
        empty_image_output: str,
        mask_color: COLOR_TYPE,
    ) -> None:
        self.image_input: str = image_input
        self.image_output: str = image_output
        self.mask_input: str = mask_input
        self.mask_output: str = mask_output
        self.empty_image_output: str = empty_image_output
        self.color: COLOR_TYPE = mask_color
        self.create_output_dirs()

    def create_output_dirs(self):
        os.makedirs(self.mask_output, exist_ok=True)
        os.makedirs(self.image_output, exist_ok=True)
        os.makedirs(self.empty_image_output, exist_ok=True)

    def color_check(self, image: Image) -> bool:
        colors = {col for _, col in image.getcolors()}
        return tuple(self.color) in colors

    def get_binary_mask(self, image: Image) -> np.ndarray:
        data = np.asarray(image)
        return np.where(np.equal(data, self.color).all(axis=2), 1, 0)

    def process_image(self, mask_path: str):
        img = Image.open(mask_path)
        mask_name = osp.basename(mask_path)
        img_name = osp.splitext(osp.basename(mask_path))[0] + ".jpg"

        if self.color_check(img):
            binary_mask = Image.fromarray(self.get_binary_mask(img))
            binary_mask.save(
                osp.splitext(osp.join(self.mask_output, mask_name))[0] + ".png"
            )
            shutil.copy(
                osp.join(self.image_input, img_name),
                self.image_output,
            )
        else:
            shutil.copy(
                osp.join(self.image_input, img_name),
                self.empty_image_output,
            )

    def process_folder(self):
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.process_image, mask_path)
                for mask_path in glob(self.mask_input + r"\*.bmp")
            ]
            for _ in tqdm(
                as_completed(futures), total=len(futures), desc="processing images"
            ):  # making progressbar
                pass


if __name__ == "__main__":
    params = params_show()
    trainval_preprocess = DataPreprocess(
        settings.IMAGE_INPUT_FOLDER,
        settings.MASK_INPUT_FOLDER,
        settings.IMAGE_OUTPUT_FOLDER,
        settings.MASK_OUTPUT_FOLDER,
        settings.EMPTY_IMAGE_OUTPUT_FOLDER,
        **params["preprocessing"]
    )
    trainval_preprocess.process_folder()

    test_preprocess = DataPreprocess(
        settings.TEST_IMAGE_INPUT_FOLDER,
        settings.TEST_MASK_INPUT_FOLDER,
        settings.TEST_IMAGE_OUTPUT_FOLDER,
        settings.TEST_MASK_OUTPUT_FOLDER,
        settings.TEST_EMPTY_IMAGE_OUTPUT_FOLDER,
        **params["preprocessing"]
    )
    test_preprocess.process_folder()
