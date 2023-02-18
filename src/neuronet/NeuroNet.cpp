#include "NeuroNet.h"
#include "./exceptions/NetworkStructureException.h"

namespace neuronet {
    NeuroNet::NeuroNet(InputLayer& input_layer_, std::vector<HiddenLayer>& hidden_layers_, OuptutLayer& ouput_layer_) {
        //Transform input_layer_ into a vector of shared pointers
        for(InputNeuron& n : input_layer_) {
            input_layer.push_back(std::make_shared<InputNeuron>(std::move(n)));
        }
        //Transform hidden_layers_ into a vector of vectors shared pointers
        for(std::vector<HiddenNeuron>& vec_n : hidden_layers_) {
            HiddenLayerP single_hidden_layer = {};
            for(HiddenNeuron& n : vec_n) {
                single_hidden_layer.push_back(std::make_shared<HiddenNeuron>(std::move(n)));
            }
            hidden_layer.push_back(single_hidden_layer);
        }
        //Transform output_layer_ into a vector of shared pointers
        for(OutputNeuron& n : ouput_layer_) {
            ouput_layer.push_back(std::make_shared<OutputNeuron>(std::move(n)));
        }
    }

    void NeuroNet::evaluate(std::vector<float>& input_values) {
        if(input_values.size() != input_layer.size()) {
            throw std::runtime_error("Input value vector size must match with the amount of input neurons in the network");
        }
        //Set input neuron values
        size_t input_values_count = input_values.size();
        for(size_t i = 0; i < input_values_count; ++i) {
            input_layer.at(i)->setValue(input_values.at(i));
        }
        //Make input neurons fire
        for(std::shared_ptr<InputNeuron>& input_n : input_layer) {
            input_n->fire();
        }
        //Make hidden neurons fire
        for(std::vector<std::shared_ptr<HiddenNeuron>>& hidden_n_layer : hidden_layer) {
            for(std::shared_ptr<HiddenNeuron>& hidden_n : hidden_n_layer) {
                hidden_n->fire();
            }
        }
        //Make ouput neurons fire
        for(std::shared_ptr<OutputNeuron>& output_n : ouput_layer) {
            output_n->fire();
        }
    }

    void NeuroNet::mesh() {
        //Sanity check
        if(input_layer.size() == 0) {
            throw exception::MissingInputLayer();
            return;
        }
        if(hidden_layer.size() == 0) {
            throw exception::MissingInputLayer();
        }
        if(hidden_layer.at(0).size() == 0) {
            throw exception::MissingInputLayer();
        }
        if(ouput_layer.size() == 0) {
            throw exception::MissingInputLayer();
        }
        mesh_input_layer();
        mesh_hidden_layer();
        mesh_output_layer();
    }

    void NeuroNet::mesh_input_layer() {
        //Mesh input layer with first hidden layer
        HiddenLayerP& fist_hidden_layer = hidden_layer.at(0);
        for(std::shared_ptr<InputNeuron>& source_n : input_layer) {
            for(std::shared_ptr<HiddenNeuron>& dest_n : fist_hidden_layer) {
                //Generate a new neuron edge
                NeuronEdge connector = NeuronEdge{source_n->getShared(), dest_n->getShared()};
                //Create a shared pointer to the edge
                std::shared_ptr<NeuronEdge> connector_pointer = std::make_shared<NeuronEdge>(std::move(connector));
                //Add the neuron edge to input/hidden layer neuron
                source_n->addOutgoingEdge(connector_pointer);
                dest_n->addIncomingEdge(connector_pointer);
            }
        }
    }
    void NeuroNet::mesh_hidden_layer() {
        //Mesh all hidden layers
        auto previous_hidden_layer_it = hidden_layer.begin(); 
        for(auto current_hidden_layer_it = ++hidden_layer.begin(); current_hidden_layer_it != hidden_layer.end(); current_hidden_layer_it++) {
            //References to the two connecting hidden layers
            HiddenLayerP& previous_hidden_layer = *previous_hidden_layer_it;
            HiddenLayerP& current_hidden_layer = *current_hidden_layer_it;
            for(std::shared_ptr<HiddenNeuron>& source_n : previous_hidden_layer) {
                for(std::shared_ptr<HiddenNeuron>& dest_n : current_hidden_layer) {
                    //Generate a new neuron edge
                    NeuronEdge connector = NeuronEdge{source_n->getShared(), dest_n->getShared()};
                    //Create a shared pointer to the edge
                    std::shared_ptr<NeuronEdge> connector_pointer = std::make_shared<NeuronEdge>(std::move(connector));
                    //Add the neuron edge to input/hidden layer neuron
                    source_n->addOutgoingEdge(connector_pointer);
                    dest_n->addIncomingEdge(connector_pointer);
                }
            }
        }
    }
    void NeuroNet::mesh_output_layer() {
        //Mesh last hidden layer with output layer
        HiddenLayerP& last_hidden_layer = hidden_layer.at(hidden_layer.size() - 1);
        for(std::shared_ptr<HiddenNeuron>& source_n : last_hidden_layer) {
            for(std::shared_ptr<OutputNeuron>& dest_n : ouput_layer) {
                //Generate a new neuron edge
                NeuronEdge connector = NeuronEdge{source_n->getShared(), dest_n->getShared()};
                //Create a shared pointer to the edge
                std::shared_ptr<NeuronEdge> connector_pointer = std::make_shared<NeuronEdge>(std::move(connector));
                //Add the neuron edge to input/hidden layer neuron
                source_n->addOutgoingEdge(connector_pointer);
                dest_n->addIncomingEdge(connector_pointer);
            }
        }
    }
}