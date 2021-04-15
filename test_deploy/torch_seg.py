import os 
import torch 
import time
import numpy as np
from torch.cuda.amp import autocast as autocast

from utils import get_path_with_annotation,preprocess_image,postprocess
from model import unet_2d


AMP_FLAG = True

s_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

csv_path = 'test.csv'
weight_path = 'unet_bladder.pth'

data_list = get_path_with_annotation(csv_path,'path','Bladder')
print(len(data_list))


model = unet_2d(n_channels=1, n_classes=2)
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

