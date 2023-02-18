#include <iostream>
#include "neuronet/NeuroNet.h"

using namespace neuronet;

int main(int argc, char const *argv[])
{   
    //Generate input layer
    InputLayer input {};
    for(int i = 0; i < 5; ++i) {
        input.push_back(InputNeuron{});
    }
    //Generate hidden layers
    std::vector<HiddenLayer> hidden_layers {};
    for(int i = 0; i < 1; ++i) {
        std::vector<HiddenNeuron> hidden_single {};
        for(int j = 0; i < 10; ++i) {
            hidden_single.push_back(HiddenNeuron{});
        }
        hidden_layers.push_back(hidden_single);
    }
    //Generate ouput layer
    OuptutLayer output {};
    for(int i = 0; i < 1; ++i) {
        output.push_back(OutputNeuron{});
    }
    //Create network
    NeuroNet network {input,hidden_layers,output};
    //Mesh the network
    network.mesh();
    //Create test input data
    std::vector<float> test_data = {};
    for(int i = 0; i < 5; ++i) {
        test_data.push_back(1);
    }
    //Evaluate the network
    network.evaluate(test_data);
    return 0;
}
