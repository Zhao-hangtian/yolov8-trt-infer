#!/bin/bash

# 检查是否存在 trt 文件
if [ -f "trt" ]; then
    echo "Found existing 'trt' file. Deleting..."
    # 删除 trt 文件
    rm trt
    echo "'trt' file deleted."
else
    echo "No existing 'trt' file found. Skipping deletion."
fi

# 编译 trt.cpp
echo "Compiling trt.cpp..."
g++ trt.cpp -o trt $(pkg-config --cflags --libs opencv4) -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvinfer -lcudart -lnvonnxparser

# 检查是否成功生成 trt 可执行文件
if [ -f "trt" ]; then
    echo "Compilation successful. Executing 'trt'..."
    # 执行 trt
    ./trt
else
    echo "Failed to compile trt.cpp"
    exit 1
fi
