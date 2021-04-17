import os 
import pandas as pd 
import h5py
import numpy as np

import torch
import torchvision.transforms as tr
from torch.nn import functional as F
from transformer import Trunc_and_Normalize,CropResize,To_Tensor

from torch.utils.data import Dataset




class DataIterator(object):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - data_len: the total number of the data
    '''
    def __init__(self, path_list, batch_size=1, roi_number=1, data_len=10000):

        self.path_list = [path_list[i%len(path_list)] for i in range(data_len)]
        self.batch_size = batch_size
        self.roi_number = roi_number
    


    def __iter__(self):
        self.current_batch = 0
        return self


    def __next__(self):

        if self.current_batch * self.batch_size < len(self.path_list):
            if (self.current_batch + 1) * self.batch_size < len(self.path_list):
                end_index = (self.current_batch + 1) * self.batch_size
            else:
                end_index = len(self.path_list)

            image_list = []
            label_list = []
            
            for item in self.path_list[self.current_batch * self.batch_size:end_index]:
                image = hdf5_reader(item,'image')
                label = hdf5_reader(item,'label')

                transforms = tr.Compose([
                    Trunc_and_Normalize(scale=[-200,400]),
                ])

                image = transforms(image)

                if self.roi_number is not None:
                    label = (label==self.roi_number).astype(np.float32) #H x W

                image = np.expand_dims(image,axis=0)
                image_list.append(image)
                label_list.append(label)

            images = np.stack(image_list,axis=0)
            labels = np.stack(label_list,axis=0)

            sample = {
                'image':torch.from_numpy(images), #n,c,h,w
                'label':torch.from_numpy(labels)  #n,h,w
            }

            self.current_batch += 1

            return sample
        else:
            raise StopIteration




class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - data_len: the total number of the data
    '''
    def __init__(self, path_list, roi_number=1, data_len=10000):

        self.path_list = [path_list[i%len(path_list)] for i in range(data_len)]
        self.roi_number = roi_number


    def __len__(self):
        return len(self.path_list)


    def __getitem__(self,index):
        image = hdf5_reader(self.path_list[index],'image')
        label = hdf5_reader(self.path_list[index],'label')

        transforms = tr.Compose([
            Trunc_and_Normalize(scale=[-200,400]),
        ])

        image = transforms(image)

        if self.roi_number is not None:
            label = (label==self.roi_number).astype(np.float32) #H x W

        image = np.expand_dims(image,axis=0)

        sample = {
            'image':torch.from_numpy(image), #c,h,w
            'label':torch.from_numpy(label)  #h,w
        }

        return sample


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

    return batch_data, torch.unsqueeze(torch.from_numpy(target),0)


def postprocess(output_data, target):
    output = F.softmax(output_data, dim=1) #n,c,h,w
    output = torch.argmax(output, 1) #n,h,w
    output = output.detach().cpu().numpy() 
    target = target.detach().cpu().numpy() 
    total_dice = 0.
    for i in range(output.shape[0]):
        dice = dice_coef(target[i],output[i])
        # print('sum: %d, dice: %.4f' % (np.sum(output),dice))
        total_dice += dice

    return total_dice / (output.shape[0]) 