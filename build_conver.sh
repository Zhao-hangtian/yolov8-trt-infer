g++ convert.cpp -o convert $(pkg-config --cflags --libs opencv4) -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvinfer -lcudart -lnvonnxparser