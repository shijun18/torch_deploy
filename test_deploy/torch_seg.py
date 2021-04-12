import os 
import torch 
import time
import numpy as np
from torch.cuda.amp import autocast as autocast

from utils import get_path_with_annotation
from model import unet_2d
from convert_to_onnx import preprocess_image,postprocess

AMP_FLAG = False

s_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

csv_path = '/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/cervical_test.csv'
weight_path = '/staff/shijun/torch_projects/Med_Seg/seg/ckpt/Cervical/2d_clean/v1.3/Bladder/fold1/epoch:127-train_loss:0.10719-train_dice:0.93932-val_loss:0.09868-val_dice:0.94423.pth'

data_list = get_path_with_annotation(csv_path,'path','Bladder')
print(len(data_list))


model = unet_2d(n_channels=1, n_classes=2, bilinear=True)
checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
model.cuda()

dice_list = []


for item in data_list:
    img, lab = preprocess_image(item)
    img = img.cuda()
    with autocast(enabled=AMP_FLAG):
        output = model(img)
    dice = postprocess(output,lab)
    dice_list.append(dice)

print('run time: %.3f' % (time.time() - s_time))
print('ave dice: %.4f' % np.mean(dice_list))

