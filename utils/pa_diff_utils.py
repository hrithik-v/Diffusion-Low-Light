from torchvision.transforms import ToPILImage

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    # ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # if img_np.shape[0] == 1:
    #     ar = ar[0]
    # else:
    #     assert img_np.shape[0] == 3, img_np.shape
    #     ar = ar.transpose(1, 2, 0)

    # return Image.fromarray(ar)
    return ToPILImage()(img_np)
    

def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    # return img_var.detach().cpu().numpy()[0]
    return img_var.detach().cpu().numpy().transpose(1, 2, 0) 