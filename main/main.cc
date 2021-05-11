#include <iostream>

#include <inference_engine.hpp>

void not_even_called_segfault() {
    InferenceEngine::TensorDesc desc;
    desc.setPrecision(InferenceEngine::Precision::FP32);
    desc.setDims({1, 10});
    auto blob = InferenceEngine::make_shared_blob<float>(desc);
}

int main()
{
    std::cout << "1" << std::endl;
    InferenceEngine::Core ie;
    std::cout << "2" << std::endl;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("/model/squeezenet1.1.xml");
    std::cout << "3" << std::endl;
    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");

    std::cout << "SUCCESS" << std::endl;
    return 0;
}
