import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):
        # Automatically resize if smaller than crop size
        # self.size is (height, width)
        h, w = self.size if isinstance(self.size, (list, tuple)) else (self.size, self.size)
        if image.width < w or image.height < h:
            image = image.resize((max(w, image.width), max(h, image.height)), Image.BICUBIC)
            label = label.resize((max(w, label.width), max(h, label.height)), Image.BICUBIC)

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)


class PairResize:
    """
    Resize both image and label to a given size.
    """
    def __init__(self, size, interpolation=2):  # 2 = BILINEAR in PIL
        self.size = size
        # Use BILINEAR interpolation as default (matching PIL.Image.BILINEAR)
        self.interpolation = interpolation

    def __call__(self, image, label):
        # Explicitly use the numerical value for interpolation 
        return F.resize(image, self.size, self.interpolation), \
               F.resize(label, self.size, self.interpolation)
