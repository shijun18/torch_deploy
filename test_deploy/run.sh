#!/bin/bash

echo 'Run Starting!!!'

mycount=0

while (($mycount  < 10))

do
    # python torch_seg.py
    # python onnx_seg.py
    python trt_seg.py
    ((mycount++))
    echo $mycount

done

echo 'End'


