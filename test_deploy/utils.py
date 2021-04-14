import os 
import pandas as pd 
import h5py
import numpy as np

import torch
import torchvision.transforms as tr
from torch.nn import functional as F
from transformer import Trunc_and_Normalize,CropResize,To_Tensor

def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image



def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)



def preprocess_image(img_path):
    # transformations for the input data
    transforms = tr.Compose([
        Trunc_and_Normalize(scale=[-200,400]),
        # CropResize(dim=(512,512)),
        To_Tensor(),
    ])

    # read input image
    input_img = hdf5_reader(img_path,'image')
    target = hdf5_reader(img_path,'label')

    # get target mask
    target = (target == 1).astype(np.float32)
    # do transformations
    input_data = transforms(input_img)
    # prepare batch
    batch_data = torch.unsqueeze(input_data, 0)

    return batch_data, target


def postprocess(output_data, target):
    output = F.softmax(output_data, dim=1)
    output = torch.squeeze(torch.argmax(output, 1))
    output = output.detach().cpu().numpy() 
    dice = dice_coef(target,output)
    print('sum: %d, dice: %.4f' % (np.sum(output),dice))
    return dice 