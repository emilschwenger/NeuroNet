#pragma once
//Standard library
#include <array>
#include <vector>
#include <stdexcept>
#include <utility>
#include <fstream>
#include <string>
#include <iterator>
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

    using InputData = std::vector<float>;
    using OutputData = std::vector<float>;
    using TrainingSet = std::pair<InputData,OutputData>;
    using TrainingData = std::vector<TrainingSet>;

    struct NeuroNet {

        //Constructor
        explicit NeuroNet(InputLayer& input_layer_, std::vector<HiddenLayer>& hidden_layers_, OuptutLayer& ouput_layer_, TrainingData training_data_);
        //Mesh the different layers int the neuronal net
        void mesh();
        //Eveluates the network with a given input vector that has the same size as input_layer
        void evaluate(const InputData&);
        OutputData evaluate_return(const InputData&);
        //start trainging with data
        void train_net(const TrainingData&);
        //start trainging with data
        void train_net_convergence(const TrainingData&);
        //export graph to file for https://graphonline.ru/en/create_graph_by_edge_list
        void export_graph_to_file(std::string path);
        //Net precision getter/setter
        void setMinNetPrecision(float precision);
        float getMinNetPrecision() const;
    private:
        //Mesh the input layer with the first hidden layer
        void mesh_input_layer();
        //Mesh all hidden layer/s together
        void mesh_hidden_layer();
        //Mesh last hidden layer with output layer together
        void mesh_output_layer();
        //Calculate error
        void propagate_error_calculation(const OutputData&);
        //Evaluate and calculate error
        void evaluate_and_propagate_error_calculation(const TrainingSet&);
        //Change weights
        void propagate_weight_change();
        /*
        * Calculates mean squared error of the ouput layer
        */
        float calculate_mean_squared_error();
    private:
        InputLayerP input_layer;
        std::vector<HiddenLayerP> hidden_layer;
        OuptutLayerP output_layer;
        //stores the bias for each layer
        std::vector<std::shared_ptr<float>> layer_bias;
        //Training data
        TrainingData training_data;
        //Net precision
        float MIN_NET_PRECISION = 0.00005f;
    };
}