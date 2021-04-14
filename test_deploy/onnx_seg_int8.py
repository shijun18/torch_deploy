import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import time

from utils import get_path_with_annotation,preprocess_image, postprocess
from calibrator import UNETEntropyCalibrator



# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path,engine_file_path=None,save_engine=False, calib=None):
    # initialize TensorRT engine and parse ONNX model
    if engine_file_path is not None and os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, \
            trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())


    else:
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as  builder, builder.create_network(explicit_batch) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # allow TensorRT to use up to 1GB of GPU memory for tactic selection
            builder.max_workspace_size = 1 << 30
            # we have only one image in batch
            builder.max_batch_size = 1
            builder.int8_mode = True

            # config    
            # profile = builder.create_optimization_profile()     
            # profile.set_shape("input", (1, 1, 512, 512),(1, 1, 512, 512),(1, 1, 512, 512))
            # config.add_optimization_profile(profile)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib


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
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")

            if save_engine:  #save engine
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())  

    context = engine.create_execution_context()

    return engine,context


def main():
    s_time = time.time()

    csv_path = 'test.csv'
    # ONNX_FILE_PATH = "unet_bladder_trt_int8.onnx"
    ONNX_FILE_PATH = "unet_bladder.onnx"

    data_list = get_path_with_annotation(csv_path,'path','Bladder')
    # initialize TensorRT engine and parse ONNX model
    calibration_cache = "unet_calibration.cache"
    calib = UNETEntropyCalibrator(data_list[:128], cache_file=calibration_cache,batch_size=32)
    # engine, context = build_engine(ONNX_FILE_PATH,'unet_bladder_int8.trt',save_engine=True,calib=calib)
    engine, context = build_engine(ONNX_FILE_PATH,calib=calib)
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    dice_list = []
    # preprocess input data

    for item in data_list:
        img, lab = preprocess_image(item)
        host_input = np.array(img, dtype=np.float32, order='C')
        cuda.memcpy_htod_async(device_input, host_input, stream)

        # run inference
        context.execute_async(batch_size=1,bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()

        # postprocess results
        
        output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, 2, 512, 512)
        dice = postprocess(output_data,lab)
        dice_list.append(dice)
    
    print('run time: %.3f' % (time.time() - s_time))
    print('ave dice: %.4f' % np.mean(dice_list))



if __name__ == '__main__':

    main()
