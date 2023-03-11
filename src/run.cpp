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
    for(int i = 0; i < 2; ++i) {
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
    
    //Create training data - logical or
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

    //Possibility to set the min net precision
    network.setMinNetPrecision(0.00000000005);

    //Evaluate the network
    network.train_net_convergence(td);

    //network.export_graph_to_file("");
    std::cout << "Training Result" << std::endl;
    
    std::cout << "Net output for input 1,0" << std::endl;
    OutputData output_1_0_or = network.evaluate_return(id1);
    std::cout << output_1_0_or.at(0) << std::endl;

    std::cout << "Net output for input 0,1" << std::endl;
    OutputData output_0_1_or = network.evaluate_return(id2);
    std::cout << output_0_1_or.at(0) << std::endl;

    std::cout << "Net output for input 0,0" << std::endl;
    OutputData output_0_0_or = network.evaluate_return(id3);
    std::cout << output_0_0_or.at(0) << std::endl;

    std::cout << "Net output for input 1,1" << std::endl;
    OutputData output_1_1_or = network.evaluate_return(id4);
    std::cout << output_1_1_or.at(0) << std::endl;

    return 0;
}
