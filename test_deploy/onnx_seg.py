import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import time
from torch.nn import functional as F
from utils import get_path_with_annotation,postprocess,DataIterator,DataGenerator
from multiprocessing import Pool
from torch.utils.data import DataLoader

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
DATA_LEN = 1000
BATCH_SIZE = 1

def build_engine(onnx_file_path,engine_file_path=None,save_engine=True):
    # initialize TensorRT engine and parse ONNX model
    if  engine_file_path is not None and os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, \
            trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())


    else:
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as  builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # allow TensorRT to use up to 1GB of GPU memory for tactic selection
            builder.max_workspace_size = 1 << 30
            # we have only one image in batch
            builder.max_batch_size = BATCH_SIZE
            # use FP16 mode if possible
            if builder.platform_has_fast_fp16:
                builder.fp16_mode = True
            # config    
            # config = builder.create_builder_config()
            # profile = builder.create_optimization_profile()     
            # profile.set_shape("input", (1, 1, 512, 512),(1, 1, 512, 512),(1, 1, 512, 512))
            # config.add_optimization_profile(profile)

            # parse ONNX
            # with open(onnx_file_path, 'rb') as model:
            #     print('Beginning ONNX file parsing')
            #     parser.parse(model.read())
            if parser.parse_from_file(onnx_file_path): 
                print('Completed parsing of ONNX file')
            else:
                raise ValueError('model parser failed!')

            last_layer = network.get_layer(network.num_layers - 1)
            # Check if last layer recognizes it's output
            if not last_layer.get_output(0):
                # If not, then mark the output using TensorRT API
                network.mark_output(last_layer.get_output(0))
            
            print(network.get_layer(network.num_layers-1).get_output(0).shape)
            # generate TensorRT engine optimized for the target platform
            print('Building an engine...')
            # if use config
            # engine = builder.build_engine(network, config)
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if save_engine:  #save engine
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())  

    context = engine.create_execution_context()

    return engine,context


def main():
    s_time = time.time()

    csv_path = 'test.csv'
    ONNX_FILE_PATH = "unet_bladder.onnx"
    # ONNX_FILE_PATH = "unet_bladder_bs4.onnx"
    # ONNX_FILE_PATH = "unet_bladder_bs8.onnx"

    # ENGINE_FILE_PATH = './v100/unet_bladder_fp16.trt'
    # ENGINE_FILE_PATH = './v100/unet_bladder_fp16_bs4.trt'
    # ENGINE_FILE_PATH = './v100/unet_bladder_fp16_bs8.trt'

    ENGINE_FILE_PATH = './p40/unet_bladder_fp16_p40.trt'
    # ENGINE_FILE_PATH = './p40/unet_bladder_fp16_bs4_p40.trt'
    # ENGINE_FILE_PATH = './p40/unet_bladder_fp16_bs8_p40.trt'


    data_list = get_path_with_annotation(csv_path,'path','Bladder')
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH,ENGINE_FILE_PATH)

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            # input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            # host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    dice_list = []
    # preprocess input data

    dataset = DataGenerator(path_list=data_list,roi_number=1,data_len=DATA_LEN)
    data_loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=2
                        )
    print(len(data_loader))

    # dataset = DataIterator(path_list=data_list,batch_size=1,roi_number=1,data_len=DATA_LEN)
    # data_loader = iter(dataset)

    tmp_total_time = 0
    post_total_time = 0
    for sample in data_loader:
        img = sample['image']
        lab = sample['label']

        tmp_time = time.time()
        host_input = np.array(img, dtype=np.float32, order='C')
        # print(host_input.shape)
        cuda.memcpy_htod_async(device_input, host_input, stream)

        # run inference
        context.execute_async(batch_size=BATCH_SIZE, bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()
        tmp_total_time += time.time() - tmp_time
        # postprocess results
        
        output_data = torch.Tensor(host_output).reshape(BATCH_SIZE, 2, 512, 512)
        
        # A torch
        # output = F.softmax(output_data, dim=1) #n,c,h,w
        # post_time = time.time()
        # output = torch.argmax(output, 1).detach().cpu().numpy() #n,h,w
        # post_total_time += time.time() - post_time
        
        # B numpy
        output_1 = F.softmax(output_data, dim=1) #n,c,h,w
        # post_time = time.time()
        output_1 = np.argmax(output_1.detach().cpu().numpy(), 1) #n,h,w
        # post_total_time += time.time() - post_time

        # A == B, but A is time-consuming 
        # assert (output == output_1).all()

        # post_time = time.time()
        # dice = postprocess(output_data,lab)
        # post_total_time += time.time() - post_time
        # dice_list.append(dice)
    
    total_time = time.time() - s_time
    print('run time: %.3f' % total_time)
    # print('post time: %.3f' % post_total_time)
    print('real run time: %.3f' % tmp_total_time)
    # print('ave dice: %.4f' % np.mean(dice_list))
    print('fps: %.3f' %(DATA_LEN/total_time))
    print('real fps: %.3f' %(DATA_LEN/tmp_total_time))


if __name__ == '__main__':

    main()
