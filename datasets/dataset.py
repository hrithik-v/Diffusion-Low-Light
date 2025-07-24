import os
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor, PairResize
import glob
import math
# import torchvision
from torchvision.transforms.functional import pad, resize

class PairPadToMultiple:
    def __init__(self, multiple=32):
        self.multiple = multiple

    def __call__(self, image, label):
        w, h = image.size
        # Compute new dimensions that are multiples of 32
        new_h = int(self.multiple * math.ceil(h / self.multiple))
        new_w = int(self.multiple * math.ceil(w / self.multiple))
        pad_h = new_h - h
        pad_w = new_w - w
        # Pad is (left, top, right, bottom)
        image = pad(image, (0, 0, pad_w, pad_h), fill=0)
        label = pad(label, (0, 0, pad_w, pad_h), fill=0)
        return image, label

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self, parse_patches=True):
        # Use separate directories for train and val based on config
        train_root = os.path.join(self.config.data.data_dir, self.config.data.train_dataset)
        val_root   = os.path.join(self.config.data.data_dir, self.config.data.val_dataset)
        # Create train and validation sets
        train_dataset = AllWeatherDataset(root_dir=train_root,
                                          patch_size=self.config.data.patch_size,
                                          train=True)
        val_dataset   = AllWeatherDataset(root_dir=val_root,
                                          patch_size=self.config.data.patch_size,
                                          train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.training.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)
        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, patch_size, train=True):
        super().__init__()
        self.root_dir = root_dir
        self.train = train
        self.patch_size = patch_size

        if self.train:
            self.raw_dir = os.path.join(root_dir, 'train', 'raw')
            self.ref_dir = os.path.join(root_dir, 'train', 'ref')
        else:
            # For validation or test phase
            self.raw_dir = os.path.join(root_dir, 'val', 'raw')
            self.ref_dir = os.path.join(root_dir, 'val', 'ref')

        # Get list of raw images
        self.input_names = sorted(glob.glob(os.path.join(self.raw_dir, '*.png')))
        print("Found raw images:", len(self.input_names), "in", self.raw_dir)

        # For each raw image, assume ref image with the same name exists in ref_dir
        # We'll match them by filename
        self.gt_names = [os.path.join(self.ref_dir, os.path.basename(x)) for x in self.input_names]

        print("Found ref images:", len(self.gt_names), "in", self.ref_dir)

        if self.train:
            self.transforms = PairCompose([
                PairPadToMultiple(32),          # Ensure dimensions are multiples of 32
                PairRandomCrop(self.patch_size),# Now safely crop to 256×256 (or whatever size you choose)
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairPadToMultiple(32),
                PairResize(self.patch_size),    # Resize to 256x256 for validation
                PairToTensor()
            ])

    def __getitem__(self, index):
        input_path = self.input_names[index]
        gt_path = self.gt_names[index]
        # print(f" number of images in input path and gt path: {len(input_path)}, {len(gt_path)}")   
        input_img = Image.open(input_path).convert('RGB')

        gt_img = Image.open(gt_path).convert('RGB')
        # print(f"input_img: {input_img.size}, gt_img: {gt_img.size}")
        # print(f"length of input_img: {len(input_img.size)}, length of gt_img: {len(gt_img.size)}")
        input_img, gt_img = self.transforms(input_img, gt_img)
        # print(f"input_img: {input_img.size()}, gt_img: {gt_img.size()}")
        # Concatenate input and ground-truth along channel dimension:
        return torch.cat([input_img, gt_img], dim=0), os.path.splitext(os.path.basename(input_path))[0]

    def __len__(self):
        return len(self.input_names)