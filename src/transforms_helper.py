import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    compose = A.Compose(
        [
            A.Resize(224, 224),
            # A.ShiftScaleRotate(
            #     shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5
            # ),
            # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return compose


if __name__ == "__main__":
    transforms = get_transforms()
    A.save(transforms, "transforms.yaml", data_format="yaml")
