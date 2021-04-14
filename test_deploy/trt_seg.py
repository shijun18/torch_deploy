import os 
import torch 
import time
import numpy as np

from utils import get_path_with_annotation,preprocess_image,postprocess
from model import unet_2d

from torch2trt import TRTModule


s_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

csv_path = 'test.csv'
# weight_path = 'unet_bladder_trt_int8.pth'
weight_path = "unet_bladder_trt_cali_int8.pth"

data_list = get_path_with_annotation(csv_path,'path','Bladder')
print(len(data_list))


model_trt = TRTModule()
model_trt.load_state_dict(torch.load(weight_path))

model_trt.eval()
model_trt.cuda()

dice_list = []


for item in data_list:
    img, lab = preprocess_image(item)
    img = img.cuda()

    output = model_trt(img)
    dice = postprocess(output,lab)
    dice_list.append(dice)

print('run time: %.3f' % (time.time() - s_time))
print('ave dice: %.4f' % np.mean(dice_list))

