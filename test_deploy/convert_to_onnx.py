import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import cv2
import onnx
import torch
import torchvision.transforms as tr
from torch.nn import functional as F

import numpy as np

from utils import hdf5_reader,get_path_with_annotation
from transformer import Trunc_and_Normalize,CropResize,To_Tensor
from model import unet_2d


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
    print('sum: %d' % np.sum(output))
    dice = dice_coef(target,output)
    print('dice: %.4f' % dice)
    return dice 


def main():
    # load pre-trained model -------------------------------------------------------------------------------------------
    
    csv_path = '/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/cervical_test.csv'
    weight_path = '/staff/shijun/torch_projects/Med_Seg/seg/ckpt/Cervical/2d_clean/v1.3/Bladder/fold1/epoch:127-train_loss:0.10719-train_dice:0.93932-val_loss:0.09868-val_dice:0.94423.pth'

    data_list = get_path_with_annotation(csv_path,'path','Bladder')


    model = unet_2d(n_channels=1, n_classes=2, bilinear=True)
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])

    # preprocessing stage ----------------------------------------------------------------------------------------------
    input, target = preprocess_image(data_list[0])
    input = input.cuda()

    # inference stage --------------------------------------------------------------------------------------------------
    model.eval()
    model.cuda()
    output = model(input)

    # post-processing stage --------------------------------------------------------------------------------------------
    dice = postprocess(output, target)

    # convert to ONNX --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = "unet_bladder.onnx"
    # input = torch.randn(8, 1, 512, 512, device='cuda')
    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True, opset_version=11,dynamic_axes={"input":{0:'batch_size'}, "output":{0:'batch_size'}})

    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)


if __name__ == '__main__':
    main()
