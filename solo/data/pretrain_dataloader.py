# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import random
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder

from PIL import Image


try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True

def pil_loader(path: str,path1: str) -> Image.Image:
    with open(path, "rb") as f,open(path1, "rb") as f1:
        img = Image.open(f).convert("RGB")
        img1 = Image.open(f1).convert("RGB")
    return img,img1
def default_loader(path: str,path1: str) -> Any:
    from torchvision import get_image_backend
    return pil_loader(path,path1)

def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])


def build_transform_pipeline(dataset, cfg):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    }

    mean, std = MEANS_N_STD.get(
        dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
    )

    augmentations = []
    if cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                cfg.crop_size,
                scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.color_jitter.brightness,
                        cfg.color_jitter.contrast,
                        cfg.color_jitter.saturation,
                        cfg.color_jitter.hue,
                    )
                ],
                p=cfg.color_jitter.prob,
            ),
        )

    if cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))

    if cfg.gaussian_blur.prob:
        augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))

    if cfg.solarization.prob:
        augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))

    if cfg.equalization.prob:
        augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))

    if cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))

    augmentations.append(transforms.ToTensor())
    augmentations.append(transforms.Normalize(mean=mean, std=std))

    augmentations = transforms.Compose(augmentations)
    return augmentations


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)


class DiffusionImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.root = root
        self.samples=[]
        self.transform = transform
        self.loader = default_loader
        class_names=[]
        class_id_to_name={}
        class_name_to_id={}
        for x in os.listdir(root):
            class_names.append(x)
        class_names.sort()
        t=0
        for i in class_names:
            class_id_to_name[i]=t
            class_name_to_id[t]=i
            t+=1
        for j in class_names:
            for k in os.listdir(root+'/'+j):
                self.samples.append((root+'/'+j+'/'+k,class_id_to_name[j],"/raid/biplab/phduser1/Hassan/diffimage_new/painting/"+j+"/"+k+"/"+"img","/raid/biplab/phduser1/Hassan/diffimage_new/sketch/"+j+"/"+k+"/"+"img","/raid/biplab/phduser1/Hassan/diffimage_new/clipart/"+j+"/"+k+"/"+"img","/raid/biplab/phduser1/Hassan/diffimage_new/quickdraw/"+j+"/"+k+"/"+"img"))
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        # original way of getting samples
        path, target,path1,path2,path3,path4 = self.samples[index]
        pr =random.random()
        if pr > 0.8:
            paths1 = path1
        elif pr> 0.6:
            paths1 = path2
        elif pr>0.4:
            paths1 = path3
        elif pr>0.2:
            paths1 = path4
        else:
            paths1= path
        r = random.randint(0, 3)
        if paths1!=path:
            sample,sample1 = self.loader(path,paths1+str(r)+".jpg")
        else: 
            sample,sample1 = self.loader(path,paths1)
        # t = path.split("/")
        
        # print(path,target)
        
        diffused_sample = sample1
        out = []
        # we know self.transform is FullTransformPipeline
        # sample=diffused_sample
        more_than_one_transform_pipeline = len(self.transform.transforms) > 1
        if more_than_one_transform_pipeline:
            for i, transform in enumerate(self.transform.transforms):
                if i == 0:
                    out.extend(transform(sample))
                else:
                    out.extend(transform(diffused_sample))
        else:
            #for _ in range(self.transform.num_crops - 1)
            out = [
                self.transform.transform(sample) for _ in range(self.transform.num_crops - 1)
            ] + [self.transform.transform(diffused_sample)]
            # transform = self.transform.transforms[0]
            # out = [
            #     transform(sample) 
            # ] + [transform(diffused_sample)]
        return out, target

class MultiImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.root = root
        self.samples=[]
        self.transform = transform
        self.loader = default_loader
        class_names=[]
        class_id_to_name={}
        class_name_to_id={}
        for x in os.listdir(root):
            class_names.append(x)
        class_names.sort()
        t=0
        for i in class_names:
            class_id_to_name[i]=t
            class_name_to_id[t]=i
            t+=1
        print(len(class_id_to_name.keys()))
        for j in class_names:
            for k in os.listdir(root+'/'+j):
                # self.samples.append((root+'/'+j+'/'+k,class_id_to_name[j],"/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/domainnet/domainnet-paint/train/"+j,"/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/domainnet/domainnet-sketch/train/"+j))
                self.samples.append((root+'/'+j+'/'+k,class_id_to_name[j],"/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/PACS/art_painting/train/"+j,"/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/PACS/sketch/train/"+j))
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        # original way of getting samples
        path, target,path1,path2 = self.samples[index]
        pr =random.random()
        if pr > 0.6:
            paths1 = path1
        elif pr> 0.3:
            paths1 = path2
        else:
            paths1= path
        # r = random.randint(0, 3)
        if paths1!=path:
            image_files = [f for f in os.listdir(paths1) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            index = random.random() * len(image_files)
            random_image = image_files[(int)(index)]
            sample,sample1 = self.loader(path,paths1+"/"+random_image)
        else: 
            sample,sample1 = self.loader(path,paths1)
        # t = path.split("/")
        
        # print(path,target)
        
        diffused_sample = sample1
        out = []
        # we know self.transform is FullTransformPipeline
        # sample=diffused_sample
        more_than_one_transform_pipeline = len(self.transform.transforms) > 1
        if more_than_one_transform_pipeline:
            for i, transform in enumerate(self.transform.transforms):
                if i == 0:
                    out.extend(transform(sample))
                else:
                    out.extend(transform(diffused_sample))
        else:
            #for _ in range(self.transform.num_crops - 1)
            out = [
                self.transform.transform(sample) for _ in range(self.transform.num_crops - 1)
            ] + [self.transform.transform(diffused_sample)]
            # transform = self.transform.transforms[0]
            # out = [
            #     transform(sample) 
            # ] + [transform(diffused_sample)]
        return out, target

def prepare_datasets(
    dataset: str,
    transform: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
    data_fraction: float = -1.0,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        no_labels (Optional[bool]): if the custom dataset has no labels.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            train=True,
            download=download,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            train_data_path,
            split="train+unlabeled",
            download=download,
            transform=transform,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = dataset_with_index(H5Dataset)(dataset, train_data_path, transform)
        elif data_format == "diffused":
            train_dataset = dataset_with_index(DiffusionImageFolder)(train_data_path, transform)
        # elif data_format == "multi":
        #     train_dataset = dataset_with_index(MultiImageFolder)(train_data_path, transform)
        else:
            train_dataset = dataset_with_index(ImageFolder)(train_data_path, transform)

    elif dataset == "custom":
        if data_format == "multi":
            dataset_class = MultiImageFolder
        elif no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_data_path, transform)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        from sklearn.model_selection import train_test_split

        if isinstance(train_dataset, CustomDatasetWithoutLabels):
            files = train_dataset.images
            (
                files,
                _,
            ) = train_test_split(files, train_size=data_fraction, random_state=42)
            train_dataset.images = files
        else:
            data = train_dataset.samples
            files = [f for f, _ in data]
            labels = [l for _, l in data]
            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
            train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
