import os 
import torch 
from torch.nn import functional as F
import time
import numpy as np
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader

from utils import get_path_with_annotation,postprocess,DataGenerator
from model import unet_2d


AMP_FLAG = True

DATA_LEN = 10000

s_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

csv_path = 'test.csv'
weight_path = 'unet_bladder.pth'

data_list = get_path_with_annotation(csv_path,'path','Bladder')
print(len(data_list))

dataset = DataGenerator(path_list=data_list,roi_number=1,data_len=DATA_LEN)
data_loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True
                        )

print(len(data_loader))

model = unet_2d(n_channels=1, n_classes=2)
checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
model.cuda()

dice_list = []


for sample in data_loader:
    img = sample['image']
    lab = sample['label']
    img = img.cuda()
    with autocast(enabled=AMP_FLAG):
        output = model(img)
    
    output = F.softmax(output, dim=1).detach().cpu().numpy() #n,c,h,w
    output = np.argmax(output, 1) #n,h,w

    # dice = postprocess(output,lab)
    # dice_list.append(dice)


total_time = time.time() - s_time
print('run time: %.3f' % total_time)
# print('ave dice: %.4f' % np.mean(dice_list))
print('fps: %.3f' %(DATA_LEN/total_time))

