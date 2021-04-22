import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

from utils import hdf5_reader


def normalize(image,scale=[-200,400]):
    image = image - scale[0]
    gray_range = scale[1] - scale[0]
    image[image < 0] = 0
    image[image > gray_range] = gray_range
    
    image = image / gray_range

    return image

# Returns a numpy buffer of shape (num_images, 1, 512, 512)
def load_data(data_list):
    image_list = []
    for item in data_list:
        input_img = hdf5_reader(item,'image')
        input_img = normalize(input_img)
        input_img = np.expand_dims(input_img,axis=0)
        image_list.append(input_img)
    
    image_array = np.stack(image_list,axis=0)

    return np.ascontiguousarray(image_array.astype(np.float32))


class UNETCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, data_list, cache_file, batch_size=8):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_data(data_list)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        print(batch.shape)
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
