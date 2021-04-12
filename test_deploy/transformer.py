import torch 
import numpy as np 
from skimage.transform import resize




class Trunc_and_Normalize(object):
  '''
  truncate gray scale and normalize to [0,1]
  '''
  def __init__(self, scale):
    self.scale = scale
    assert len(self.scale) == 2, 'scale error'

  def __call__(self, image):
 
      # gray truncation
      image = image - self.scale[0]
      gray_range = self.scale[1] - self.scale[0]
      image[image < 0] = 0
      image[image > gray_range] = gray_range
      
      image = image / gray_range

      return image




class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None,crop=0):
        self.dim = dim
        self.crop = crop

    def __call__(self, image):

        # image: numpy array
        # crop
        if self.crop != 0:
            image = image[self.crop:-self.crop, self.crop:-self.crop]
        # resize
        if self.dim is not None and image.shape != self.dim:
            image = resize(image, self.dim, anti_aliasing=True)

        return image


class To_Tensor(object):
    '''
    Convert the data in sample to torch Tensor.
    '''

    def __call__(self,image):
        # expand dims
        if len(image.shape) == 2:
            image = np.expand_dims(image,axis=0)
        else:
            image = np.transpose(image,(2,0,1))
        # convert to Tensor
        image = torch.from_numpy(image)
                
        return image