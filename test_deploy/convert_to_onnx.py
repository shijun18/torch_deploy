import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import onnx
import torch

import numpy as np

from utils import hdf5_reader,get_path_with_annotation,preprocess_image,postprocess
from model import unet_2d



def main():
    # load pre-trained model -------------------------------------------------------------------------------------------
    
    csv_path = 'test.csv'
    weight_path = 'unet_bladder.pth'

    data_list = get_path_with_annotation(csv_path,'path','Bladder')


    model = unet_2d(n_channels=1, n_classes=2)
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
    # dynamic
    # ONNX_FILE_PATH = "unet_bladder_dynamic.onnx"
    # ONNX_FILE_PATH = "unet_bladder.onnx"
    # ONNX_FILE_PATH = "unet_bladder_bs8.onnx"
    ONNX_FILE_PATH = "unet_bladder_bs4.onnx"
    input = torch.randn(4, 1, 512, 512, device='cuda')
    
    # torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True, opset_version=11, dynamic_axes={"input":{0:'batch_size'}, "output":{0:'batch_size'}})
    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True, opset_version=11)
    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)


if __name__ == '__main__':
    main()
