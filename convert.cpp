#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity != nvinfer1::ILogger::Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

int main() {
    Logger logger;

    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);
    // nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    // INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));



    auto parser = nvonnxparser::createParser(*network, logger);

    // Parse ONNX file
    if (!parser->parseFromFile("best.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ERROR: could not parse the model." << std::endl;
        return -1;
    }

    // Build the engine
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);  // 1MB
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Serialize the engine
    nvinfer1::IHostMemory* modelStream = engine->serialize();

    // Save the engine to file
    std::ofstream outFile("best.engine", std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    // Clean up
    modelStream->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    config->destroy();
    builder->destroy();

    return 0;
}
