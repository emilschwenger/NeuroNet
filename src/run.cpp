#include <iostream>
#include "neuronet/NeuroNet.h"

using namespace neuronet;

int main(int argc, char const *argv[])
{   
    //Generate input layer
    InputLayer input {};
    for(int i = 0; i < 2; ++i) {
        input.push_back(InputNeuron{});
    }
    //Generate hidden layers
    std::vector<HiddenLayer> hidden_layers {};
    for(int i = 0; i < 3; ++i) {
        std::vector<HiddenNeuron> hidden_single {};
        for(int j = 0; j < 2; ++j) {
            hidden_single.push_back(HiddenNeuron{});
        }
        hidden_layers.push_back(hidden_single);
    }
    //Generate ouput layer
    OuptutLayer output {};
    for(int i = 0; i < 1; ++i) {
        output.push_back(OutputNeuron{});
    }
    //Create test input data
    InputData id1 = {1,0};
    OutputData od1 = {1};
    InputData id2 = {0,1};
    OutputData od2 = {1};
    InputData id3 = {0,0};
    OutputData od3 = {0};
    InputData id4 = {1,1};
    OutputData od4 = {1};
    TrainingData td;
    td.push_back(std::make_pair(id1,od1));
    td.push_back(std::make_pair(id2,od2));
    td.push_back(std::make_pair(id3,od3));
    td.push_back(std::make_pair(id4,od4));
    //Create network
    NeuroNet network {input,hidden_layers,output,td};
    //Mesh the network
    network.mesh();
    //Evaluate the network
    for(int i = 0; i < 1; ++i) {
        network.train_net(td);
    }
    network.export_graph_to_file("/Users/emilschwenger/TUM/C++ Grundlagen/Test Workspace/NeuroNet/graph.txt");
    network.evaluate(id3);
    return 0;
}
