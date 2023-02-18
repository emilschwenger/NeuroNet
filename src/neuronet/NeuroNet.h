#pragma once
//Standard library
#include <array>
#include <vector>
#include <stdexcept>
//User defined
#include "InputNeuron.h"
#include "HiddenNeuron.h"
#include "OutputNeuron.h"
#include "NeuronEdge.h"

namespace neuronet {

    using InputLayer = std::vector<InputNeuron>;
    using HiddenLayer = std::vector<HiddenNeuron>;
    using OuptutLayer = std::vector<OutputNeuron>;

    using InputLayerP = std::vector<std::shared_ptr<InputNeuron>>;
    using HiddenLayerP = std::vector<std::shared_ptr<HiddenNeuron>>;
    using OuptutLayerP = std::vector<std::shared_ptr<OutputNeuron>>;

    struct NeuroNet {
        //Constructor
        explicit NeuroNet(InputLayer& input_layer_, std::vector<HiddenLayer>& hidden_layers_, OuptutLayer& ouput_layer_);
        //Mesh the different layers int the neuronal net
        void mesh();
        //Eveluates the network with a given input vector that has the same size as input_layer
        void evaluate(std::vector<float>&);
    private:
        //Mesh the input layer with the first hidden layer
        void mesh_input_layer();
        //Mesh all hidden layer/s together
        void mesh_hidden_layer();
        //Mesh last hidden layer with output layer together
        void mesh_output_layer();
    private:
        InputLayerP input_layer;
        std::vector<HiddenLayerP> hidden_layer;
        OuptutLayerP ouput_layer;
    };
}