import os 
import torch 
import time
import numpy as np

from utils import get_path_with_annotation,postprocess,DataGenerator
from model import unet_2d

from torch2trt import TRTModule
from torch.utils.data import DataLoader



DATA_LEN = 1000

s_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

csv_path = 'test.csv'
# weight_path = 'unet_bladder_trt_int8.pth'
# weight_path = 'unet_bladder_trt_int8_bs8_p40.pth'

# weight_path = "unet_bladder_trt_cali_int8.pth"
weight_path = "unet_bladder_trt_cali_int8_bs8_p40.pth"

# weight_path = "unet_bladder_trt_fp16.pth"
# weight_path = "unet_bladder_trt_fp16_p40.pth"

data_list = get_path_with_annotation(csv_path,'path','Bladder')
print(len(data_list))


dataset = DataGenerator(path_list=data_list,roi_number=1,data_len=DATA_LEN)
data_loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=False,
                        num_workers=2
                        )

print(len(data_loader))

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(weight_path))

model_trt.eval()
model_trt.cuda()

dice_list = []

tmp_total_time = 0
for sample in data_loader:
    img = sample['image']
    lab = sample['label']
    
    tmp_time = time.time()
    img = img.cuda()
    output = model_trt(img)
    tmp_total_time += time.time() - tmp_time
    dice = postprocess(output,lab)
    dice_list.append(dice)


total_time = time.time() - s_time
print('run time: %.3f' % total_time)
print('real run time: %.3f' % tmp_total_time)
print('ave dice: %.4f' % np.mean(dice_list))
print('fps: %.3f' %(DATA_LEN/total_time))
print('real fps: %.3f' %(DATA_LEN/tmp_total_time))

