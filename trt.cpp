#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
    {
        if (severity != nvinfer1::ILogger::Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

// Load the engine file into a runtime object
nvinfer1::ICudaEngine *loadEngine(const std::string &engineFile, Logger &logger)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening engine file: " << engineFile << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t fileSize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    if (!file)
    {
        std::cerr << "Error reading engine file: " << engineFile << std::endl;
        return nullptr;
    }

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    return runtime->deserializeCudaEngine(engineData.data(), fileSize, nullptr);
}

void preprocessImage(const std::string &imagePath, float *inputData, int batchIndex, int channels, int height, int width)
{
    cv::Mat img = cv::imread(imagePath);
    cv::resize(img, img, cv::Size(width, height));
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    std::vector<cv::Mat> inputChannels(channels);
    cv::split(img, inputChannels);

    for (int i = 0; i < channels; ++i)
    {
        memcpy(inputData + batchIndex * channels * height * width + i * height * width, inputChannels[i].data, height * width * sizeof(float));
    }
}

int main()
{
    Logger logger;
    const std::string engineFile = "best.engine";

    // Load the TensorRT engine
    nvinfer1::ICudaEngine *engine = loadEngine(engineFile, logger);
    if (!engine)
    {
        std::cerr << "Failed to load the engine." << std::endl;
        return -1;
    }

    // Print input and output tensor info
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        if (engine->bindingIsInput(i))
        {
            std::cout << "Input tensor: " << engine->getBindingName(i) << ", shape: ";
        }
        else
        {
            std::cout << "Output tensor: " << engine->getBindingName(i) << ", shape: ";
        }
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j] << " ";
        }
        std::cout << std::endl;
    }

    // Assuming input tensor name is "input" and output tensor name is "output"
    const int inputIndex = engine->getBindingIndex("input");   // Replace with actual input tensor name
    const int outputIndex = engine->getBindingIndex("output"); // Replace with actual output tensor name

    // Get input and output dimensions
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

    const int batchSize = inputDims.d[0];
    const int inputChannels = inputDims.d[1];
    const int inputHeight = inputDims.d[2];
    const int inputWidth = inputDims.d[3];
    const int outputSize = batchSize * outputDims.d[1];

    // Prepare input and output buffers
    void *buffers[2];
    cudaMalloc(&buffers[inputIndex], batchSize * inputChannels * inputHeight * inputWidth * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));

    // Load and preprocess images
    std::vector<std::string> imagePaths = {
        "/usr/src/ultralytics/20240520_195712_210.png",
        "/usr/src/ultralytics/20240520_195712_210.png",
        "/usr/src/ultralytics/20240520_195712_210.png",
        "/usr/src/ultralytics/20240520_195712_210.png",
    };
    std::vector<float> inputData(batchSize * inputChannels * inputHeight * inputWidth);

    for (int i = 0; i < imagePaths.size(); ++i)
    {
        preprocessImage(imagePaths[i], inputData.data(), i, inputChannels, inputHeight, inputWidth);
    }

    cudaMemcpy(buffers[inputIndex], inputData.data(), batchSize * inputChannels * inputHeight * inputWidth * sizeof(float), cudaMemcpyHostToDevice);

    // Create an execution context
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Failed to create execution context." << std::endl;
        engine->destroy();
        return -1;
    }

    // Run inference
    context->executeV2(buffers);

    // Copy output data from device to host
    std::vector<float> outputData(outputSize);
    cudaMemcpy(outputData.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Process output data
    // Example: print output data
    for (size_t i = 0; i < outputData.size(); ++i)
    {
        std::cout << "Output[" << i << "] = " << outputData[i] << std::endl;
    }

    // Clean up
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();

    return 0;
}
