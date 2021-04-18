# Torch模型部署实践 (Python)——低精度推理

因科研需要，在过去一周，学习了torch模型部署的相关知识并付诸实践，在此简单记录学习过程。由于时间有限，该项目当前只包含`Python`版本的相关实现，后面会更新`C++`版本 (不拖延的话:stuck_out_tongue_winking_eye:)。

## 相关背景

与纯科研目的不同，在实际应用中，AI模型在训练完成后，需要部署到云端或终端设备（以及边缘设备）。因此，对模型的运行效率有较高的要求，如**实时性、低延时**。解决该问题一般有两个思路，一是开发出更轻量且满足实际需求的模型（如`MobileNet`），二是在部署时结合不同设备的特性和开发环境，实现**推理加速**。显然，这两种方法是正交的，前者的难度相对较高，而后者的可操作空间更大。**NVIDIA**的**TensorRT**是一款针对高性能深度学习推理的**SDK**，其提供用于（预先训练过的）模型推理的API（Python或C++），并针对用户平台生成优化的运行时引擎。**TensorRT**支持多种优化加速方案，如**低精度（FP16或Int8）**推理，算子融合等。本项目的第一个目标就是利用**TensorRT**提供的**低精度推理**方案实现对自定义模型的推理加速，为后期的应用部署工作打下基础。

## 软硬件环境

本项目实验所用的主要软硬件环境如下：

- **TensorRT Version**: 7.2.2.3
- **NVIDIA GPU**: Tesla V100 (16GB) / P40 (24GB) / P100 (16GB),  (*PASCAL 架构不包含Tensor Core*)
- **NVIDIA Driver Version**: 450.102.04
- **CUDA Version**: 11.0
- **CUDNN Version**: 8.0.5
- **Operating System**: ubantu 18.04
- **Python Version**: 3.7
- **PyTorch Version**: 1.7.1

*Note:* **切记不同软件的版本需要匹配（主要与CUDA匹配），在安装之前需要仔细核对，否则会导致许多麻烦！！**

## 相关软件安装

该项目当前是基于Python实现的，因此相关软件的安装相对比较简单，不过要注意与CUDA的版本对应。其中，**TensorRT**安装参考[Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-722/install-guide/index.html#installing-tar)，建议采用`Tar`解压的方式安装。其余第三库见`requirements.txt`。

## 低精度推理部署方案

为了制定实验方案，对目前Torch模型部署的通用方案进行了调研。

可知利用**TensorRT**实现**Torch**模型的低精度推理部署一般思路（**Plan A**）为：

- **Step1**: Torch 模型(**FP32**)转ONNX格式 (`.pth ` >> ` .onnx`）
- **Step2**: 利用TensorRT对ONNX模型进行低精度（**FP16或Int8**）推理部署

其中，**ONNX**（Open Neural Network Exchange）是一种机器模型的开放存储格式，无论是Torch、Tensorflow还是其他深度学习框架生成的模型，都可以转存成`.onnx`格式，然后使用TensorRT进行部署。这种中间格式存在的好处显而易见，**可以消除不同深度学习框架的差异**，TensorRT只需支持`.onnx`格式即可。

除了上述方案，也可以采取下面这个更为直接的方案（**Plan B**）：

- **Step1**: 高精度Torch模型直接转化为低精度**TRT模型**，即`fp32.pth` >> `fp16.pth` 或`int.pth`
- **Step2**: 利用TensorRT对低精度模型进行部署

**Plan B** 依赖于第三方库:`torch2trt`，生成低精度模型后，推理部署非常简单，类似于直接使用torch推理。

## 测试结果

该项目实验所用模型为`Unet`，是一个简单的单目标语义分割任务。

### FP16推理

目前Torch框架已经很好地支持了混合精度训练和推理，调用也非常简单，该方案可以作为额外的对照组：

~~~python
import torch 
from torch.cuda.amp import autocast as autocast

def preprocess(img_path):
    '''processing
    '''
    return

img_path = 'your_path_to_image'

model = your_model(*args)
model.load_state_dict(torch.load(weight_path)['state_dict'])
model.eval()
model.cuda()

img = preprocess(img_path)
img = img.cuda
with autocast(enabled=True):
    output = model(img)
~~~

#### Plan A: Torch >> ONNX >> TensorRT

##### Torch转ONNX

~~~python
#save path
ONNX_FILE_PATH = "unet.onnx"
input = torch.randn(1, 1, 512, 512, device='cuda')
torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True, opset_version=11)
onnx_model = onnx.load(ONNX_FILE_PATH)
# check that the model converted fine
onnx.checker.check_model(onnx_model)
~~~

**这里需要注意：**目前网上很多教程和示例都是围绕分类模型的转化，本项目用的是分割模型，包含了插值和填充操作，由于`torch.onnx`目前对部分算子的支持还不是很完善，所以转化的过程会遇到一些BUG，比如`Pad`操作只允许接收显示常量作为参数。刚开始为了解决这个问题将`opset_vesion`设置为**11**，虽然不再提示报错，但是生成的`.onnx`模型无法被TensorRT正常加载。重新查看相关文档，发现仍然是算子不支持的问题，最终通过移除`Pad`操作解决了该问题。**PS：**在本项目中，`Pad`操作在运行时做的是$0$填充，并没有改变数据尺寸，也就是无意义操作，所以移除后不影响原模型结构。所以，若是自定义模型中`Pad`操作确实发挥了作用，就需要将传递的参数全部修改成常量以此来解决该问题。

##### TensorRT部署ONNX

~~~python
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger()
def build_engine(onnx_file_path,engine_file_path=None,save_engine=False):
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
            builder.max_batch_size = 1
            # use FP16 mode if possible
            if builder.platform_has_fast_fp16:
                builder.fp16_mode = True

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
    ONNX_FILE_PATH = "unet.onnx"
    engine, context = build_engine(ONNX_FILE_PATH,'unet_fp16.trt',True)
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
    
    img_path = 'your_path_to_image'
    img = preprocess(img_path)
    host_input = np.array(img, dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, 2, 512, 512)
~~~

**这里需要注意两点：**

- 创建`engine`时需要显示地传入`explicit_batch`，否则会报错！
- 首次创建`engine`时速度会很慢，可以将创建后的`engine`保存成`.trt`文件，下次可直接从文件中解序列读取，可以极大缩短创建时间。

#### Plan B: Torch >> TRT >> torch2trt

##### Torch转TRT

~~~python
import torch
import tensorrt as trt
from torch2trt import torch2trt



TRT_FILE_PATH = "unet_fp16.pth" 
data = torch.randn(1, 1, 512, 512).cuda()

# FP16
model_trt = torch2trt(model, [data], fp16_mode=True)
torch.save(model_trt.state_dict(), TRT_FILE_PATH)
~~~

##### torch2trt部署

~~~python
import torch
from torch2trt import TRTModule

weight_path = "unet_fp16.pth"

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(weight_path))

model_trt.eval()
model_trt.cuda()

img_path = 'your_path_to_image'
img = preprocess(img_path)
img = img.cuda()
output = model_trt(img)
~~~

使用torch2trt进行部署的过程与直接用torch推理的过程基本一致。*注意：TRT模型虽然是通过`torch.save()`保存的，但是必须使用`TRTModule`进行实例化！！*

##### 结果分析

- **FP32 Baseline**
  - **Batch Size = 1**

    | 方案  | GPU  | #Samples | runtime (s) | FPS    |
    | ----- | ---- | -------- | ----------- | ------ |
    | torch | V100 | 10000    | 238.695     | 41.894 |
    | torch | V100 | 1000     | 38.384      | 25.895 |
    | torch | P40  | 10000    | 404.685     | 24.711 |
    | torch | P40  | 1000     | 47.508      | 21.786 |
    
  - **Batch Size = 4**
  
    | 方案  | GPU  | #Samples | runtime (s) | FPS        |
    | ----- | ---- | -------- | ----------- | ---------- |
    | torch | V100 | 10000    | 211.266     | **47.334** |
    | torch | V100 | 1000     | 33.648      | 29.719     |
    | torch | P40  | 10000    | 385.357     | 25.950     |
    | torch | P40  | 1000     | 44.456      | 22.494     |
  - **Batch Size = 8**  ps：实验所用v100显卡只有16GB显存，input shape = (1,512,512)，batch size无法设置 > 5 :sweat_smile:
  
    | 方案  | GPU  | #Samples | runtime (s) | FPS    |
    | ----- | ---- | -------- | ----------- | ------ |
    | torch | V100 | 10000    | -           | -      |
    | torch | V100 | 1000     | -           | -      |
    | torch | P40  | 10000    | 389.144     | 25.697 |
    | torch | P40  | 1000     | 44.621      | 22.411 |
  
     **观察：**增大`batch size`可以有限地降低总的推理时间，并不是线性的。**不知道具体原因，需要去了解torch调用GPU的内部机制（新坑）**
  
- **FP16 results**

  - **Batch Size = 1**

    | 方案             | GPU  | #Samples | runtime (s) | FPS        |
    | ---------------- | ---- | -------- | ----------- | ---------- |
    | torch + 混合精度 | V100 | 10000    | 151.227     | 66.126     |
    | torch + 混合精度 | V100 | 1000     | 25.705      | 38.903     |
    | torch + 混合精度 | P40  | 10000    | 474.733     | 21.064     |
    | torch + 混合精度 | P40  | 1000     | 53.330      | 18.751     |
    | onnx + TensorRT  | V100 | 10000    | 729.698     | 13.704     |
    | onnx + TensorRT  | V100 | 1000     | 82.376      | 12.021     |
    | onnx + TensorRT  | P40  | 10000    | 949.490     | 10.532     |
    | onnx + TensorRT  | P40  | 1000     | 99.546      | 10.046     |
    | torch2trt        | V100 | 10000    | **100.290** | **99.711** |
    | torch2trt        | V100 | 1000     | 33.148      | 30.168     |
    | torch2trt        | P40  | 10000    | 247.966     | 40.328     |
    | torch2trt        | P40  | 1000     | 34.430      | 29.044     |



---

### Int8推理

#### Plan A: Torch >> ONNX >> TensorRT

Torch转ONNX与FP16推理相同，区别在于部署过程。

~~~python
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger()
def build_engine(onnx_file_path,engine_file_path=None,save_engine=False, calib=None):
    # initialize TensorRT engine and parse ONNX model
    if  engine_file_path is not None and os.path.exists(engine_file_path):
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
            # use FP16 mode if possible
            builder.int8_mode = True
			
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
            
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
            #use config
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")

            if save_engine:  #save engine
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())  

    context = engine.create_execution_context()

    return engine,context

def main():
    ONNX_FILE_PATH = "unet.onnx"
    calibration_cache = "unet_calibration.cache"
    calib = UNETEntropyCalibrator(*args)
    engine, context = build_engine(ONNX_FILE_PATH,'unet_int8.trt',save_engine=True,calib=calib)
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
    
    img_path = 'your_path_to_image'
    img = preprocess(img_path)
    host_input = np.array(img, dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, 2, 512, 512)
~~~

与FP16推理不同，创建TensorRT创建Int8推理引擎需要依赖于**数据校准**，相关知识可参考[Int8推理](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)，校准器的定义如下：

~~~python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def load_data(data_list):
    '''load data as array
    '''
    return np.ascontiguousarray(array.astype(np.float32))


class UNETEntropyCalibrator(trt.IInt8MinMaxCalibrator):
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

~~~

这里采用`MinMax`校准算法，之前尝试过`Entropy`算法，但是结果有问题！**值得注意的是：**在做数据校准时，不需要所有的测试样本，抽样一部分即可，`batch_size`尽可能设大。在创建`engine`时，需要定义一个`config`进行参数配置，其余与FP16推理无差别。

#### Plan B: Torch >> TRT >> torch2trt

##### Torch转TRT

与FP16不同，创建Int8的TRT模型，有两种模式：

- 不使用真实数据校准

~~~python
import torch
import tensorrt as trt
from torch2trt import torch2trt



TRT_FILE_PATH = "unet_int8.pth" 
data = torch.randn(1, 1, 512, 512).cuda()

# no calibrator
model_trt = torch2trt(model, [data], int8_mode=True, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION,int8_calib_batch_size=32)
torch.save(model_trt.state_dict(), TRT_FILE_PATH)
~~~

- 使用真实数据校准

~~~python
import torch
import tensorrt as trt
from torch2trt import torch2trt

class ImageCalibDataset():
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, _ = preprocess_image(self.data_list[idx])
        return [image]


TRT_FILE_PATH = "unet_cali_int8.pth" 
data = torch.randn(1, 1, 512, 512).cuda()

# use calibrator
dataset = ImageCalibDataset(*args)
model_trt = torch2trt(model, [data], int8_calib_dataset=dataset, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION, int8_mode=True, int8_calib_batch_size=32)
torch.save(model_trt.state_dict(), TRT_FILE_PATH)
~~~

**注意：**不管是否使用真实数据校准，`int8_calib_algorithm`必须指定算法，在这里用的是`trt.CalibrationAlgoType.MINMAX_CALIBRATION`，不指定的话精度损失会比较大！！

##### 结果分析

- **Int8**推理
  - **Batch Size = 1**

    | 方案                   | GPU  | #Samples | runtime (s) | FPS  |
    | ---------------------- | ---- | -------- | ----------- | ---- |
    | onnx + TensorRT        | V100 | 10000    |             |      |
    | onnx + TensorRT        | V100 | 1000     |             |      |
    | onnx + TensorRT        | P40  | 10000    |             |      |
    | onnx + TensorRT        | P40  | 1000     |             |      |
    | torch2trt              | V100 | 10000    |             |      |
    | torch2trt              | V100 | 1000     |             |      |
    | torch2trt              | P40  | 10000    |             |      |
    | torch2trt              | P40  | 1000     |             |      |
    | torch2trt + calibrator | V100 | 10000    |             |      |
    | torch2trt + calibrator | V100 | 1000     |             |      |
    | torch2trt + calibrator | P40  | 10000    |             |      |
    | torch2trt + calibrator | P40  | 1000     |             |      |

:cry:*由于在实验室没法独享机器，测试的数据不准确，分析和后续测试会补充！*



