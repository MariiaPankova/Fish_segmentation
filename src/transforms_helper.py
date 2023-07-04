import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    compose = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return compose


if __name__ == "__main__":
    transforms = get_transforms()
    A.save(transforms, "transforms.yaml", data_format="yaml")