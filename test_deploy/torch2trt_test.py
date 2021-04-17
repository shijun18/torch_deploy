import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import onnx
import torch

import numpy as np

from utils import get_path_with_annotation,preprocess_image,postprocess
from model import unet_2d

import tensorrt as trt
from torch2trt import torch2trt


class ImageCalibDataset():

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, _ = preprocess_image(self.data_list[idx])
        # image = image[None, ...]  # add batch dimension
        # print(image.size())
        image = image[0]
        return [image]




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
    input = input.float()
    # inference stage --------------------------------------------------------------------------------------------------
    model = model.float()
    model.eval()
    model.cuda()
    output = model(input)

    # post-processing stage --------------------------------------------------------------------------------------------
    dice = postprocess(output, target)

    # convert to trt --------------------------------------------------------------------------------------------------
    # TRT_FILE_PATH = "unet_bladder_trt_fp16_p40.pth" #0.9236
    # TRT_FILE_PATH = "unet_bladder_trt_int8_p40.pth" #0.89
    TRT_FILE_PATH = "unet_bladder_trt_cali_int8_p40.pth" #0.9219
    data = torch.randn(1, 1, 512, 512).cuda()

    # FP16
    # model_trt = torch2trt(model, [data], fp16_mode=True)
    
    # Int 8 
    # Plan A --------------------------------------------------------------------------------------------------
    # It's OK
    # model_trt = torch2trt(model, [data], int8_mode=True, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION, int8_calib_batch_size=32)

    # Plan B --------------------------------------------------------------------------------------------------
    dataset = ImageCalibDataset(data_list=data_list[:128])
    model_trt = torch2trt(model, [data], int8_calib_dataset=dataset, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION, int8_mode=True, int8_calib_batch_size=32)
    
    # Note:
    # use data calibrator can improve the accuracy of the quantized model
    # Plan B is better!!
    
    torch.save(model_trt.state_dict(), TRT_FILE_PATH)

    output = model_trt(input)

    # post-processing stage --------------------------------------------------------------------------------------------
    dice = postprocess(output, target)
    
    '''Failed !!! why?

    # convert to onnx --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = "unet_bladder_trt_int8.onnx"
    torch.onnx._export(model_trt, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True, opset_version=11)
    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)
    '''

if __name__ == '__main__':
    main()
