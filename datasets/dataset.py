import os
import torch
import torch.utils.data
import PIL
from PIL import Image
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor, PairResize


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset),
                                          patch_size=self.config.data.patch_size)
        val_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset),
                                        patch_size=self.config.data.patch_size, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.raw_dir = os.path.join(dir, 'input')
        self.ref_dir = os.path.join(dir, 'GT')
        self.patch_size = patch_size

        # List all files in raw_dir (assuming all are images)
        self.input_names = sorted([f for f in os.listdir(self.raw_dir) if os.path.isfile(os.path.join(self.raw_dir, f))])
        self.gt_names = self.input_names  # Assume same filenames in ref_dir

        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            # Resize validation images to patch_size instead of cropping
            self.transforms = PairCompose([
                # PairResize((self.patch_size, self.patch_size)),
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = os.path.splitext(input_name)[0]
        input_img = Image.open(os.path.join(self.raw_dir, input_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.ref_dir, gt_name)).convert('RGB')

        # Automatically resize images if smaller than patch size
        min_size = self.patch_size
        if input_img.width < min_size or input_img.height < min_size:
            input_img = input_img.resize((max(min_size, input_img.width), max(min_size, input_img.height)), Image.BICUBIC)
        if gt_img.width < min_size or gt_img.height < min_size:
            gt_img = gt_img.resize((max(min_size, gt_img.width), max(min_size, gt_img.height)), Image.BICUBIC)

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
