import torch
import shutil
import os
import torchvision.utils as tvu


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    print("Saving checkpoint to {}".format(filename))
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Check if keys are prefixed with 'module.'
    has_module = any(k.startswith('module.') for k in state_dict.keys())
    # Check if more than one CUDA device is available
    num_cuda = torch.cuda.device_count()

    new_state_dict = {}
    if has_module and not (num_cuda > 1):
        # Remove 'module.' prefix
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k
            new_state_dict[new_key] = v
        checkpoint['state_dict'] = new_state_dict
    elif not has_module and (num_cuda > 1):
        # Add 'module.' prefix
        for k, v in state_dict.items():
            new_key = 'module.' + k if not k.startswith('module.') else k
            new_state_dict[new_key] = v
        checkpoint['state_dict'] = new_state_dict
    # else: no change needed

    return checkpoint
