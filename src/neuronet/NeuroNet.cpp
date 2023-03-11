#include "NeuroNet.h"
#include "./exceptions/NetworkStructureException.h"

namespace neuronet {

    constexpr static float BIAS = 0;

    NeuroNet::NeuroNet(InputLayer& input_layer_, std::vector<HiddenLayer>& hidden_layers_, OuptutLayer& ouput_layer_, TrainingData training_data_) 
    : layer_bias(2 + hidden_layers_.size(), std::make_unique<float>(BIAS)), training_data{training_data_} {
        int current_layer = 0;
        //Transform input_layer_ into a vector of shared pointers
        for(InputNeuron& n : input_layer_) {
            std::shared_ptr<InputNeuron> neuron_ptr = std::make_shared<InputNeuron>(std::move(n));
            neuron_ptr->setBias(layer_bias.at(current_layer));
            input_layer.push_back(neuron_ptr);
        }
        //Transform hidden_layers_ into a vector of vectors shared pointers
        for(std::vector<HiddenNeuron>& vec_n : hidden_layers_) {
            current_layer += 1;
            HiddenLayerP single_hidden_layer = {};
            for(HiddenNeuron& n : vec_n) {
                std::shared_ptr<HiddenNeuron> neuron_ptr = std::make_shared<HiddenNeuron>(std::move(n));
                neuron_ptr->setBias(layer_bias.at(current_layer));
                single_hidden_layer.push_back(neuron_ptr);
            }
            hidden_layer.push_back(single_hidden_layer);
        }

        //Transform output_layer_ into a vector of shared pointers
        current_layer += 1;
        for(OutputNeuron& n : ouput_layer_) {
            std::shared_ptr<OutputNeuron> neuron_ptr = std::make_shared<OutputNeuron>(std::move(n));
            neuron_ptr->setBias(layer_bias.at(current_layer));
            output_layer.push_back(neuron_ptr);
        }
    }

    void NeuroNet::evaluate(const InputData& input_values) {
        if(input_values.size() != input_layer.size()) {
            throw std::runtime_error("Input value vector size must match with the amount of input neurons in the network");
        }
        //Set input neuron values
        size_t input_values_count = input_values.size();
        for(size_t i = 0; i < input_values_count; ++i) {
            input_layer.at(i)->setValueIn(input_values.at(i));
            input_layer.at(i)->setValueOut(input_values.at(i));
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
        for(std::shared_ptr<OutputNeuron>& output_n : output_layer) {
            output_n->fire();
        }
    }

    OutputData NeuroNet::evaluate_return(const InputData& data) {
        evaluate(data);
        std::vector<float> net_output{};
        for(std::shared_ptr<OutputNeuron>& output_n : output_layer) {
            net_output.push_back(output_n->getValueOut());
        }
        return net_output;
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
        if(output_layer.size() == 0) {
            throw exception::MissingInputLayer();
        }
        mesh_input_layer();
        mesh_hidden_layer();
        mesh_output_layer();
    }

    float NeuroNet::calculate_mean_squared_error() {
        float squared_error = 0;
        for(std::shared_ptr<OutputNeuron>& output_n : output_layer) {
            squared_error += (output_n->getError() * output_n->getError());
        }
        return (1.0/static_cast<float>(output_layer.size())) * squared_error;
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
            //Increase iterator
            previous_hidden_layer_it++;
        }
    }
    void NeuroNet::mesh_output_layer() {
        //Mesh last hidden layer with output layer
        HiddenLayerP& last_hidden_layer = hidden_layer.at(hidden_layer.size() - 1);
        for(std::shared_ptr<HiddenNeuron>& source_n : last_hidden_layer) {
            for(std::shared_ptr<OutputNeuron>& dest_n : output_layer) {
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

    void NeuroNet::propagate_error_calculation(const OutputData& output_data_) {
        if(output_data_.size() != output_layer.size()) {
            throw std::runtime_error("Training data ouput size must be equal to output layer size");
        }
        //Set output data in neurons
        size_t data_output_index = 0;
        for(std::shared_ptr<OutputNeuron>& output_n : output_layer) {
            output_n->setExpected(output_data_.at(data_output_index));
        }
        //Make ouput neurons calculate error
        for(std::shared_ptr<OutputNeuron>& output_n : output_layer) {
            output_n->calc_error();
        }
        //Make hidden neurons calculate error in reverse order
        for(auto it = hidden_layer.rbegin(); it != hidden_layer.rend(); ++it) {
            std::vector<std::shared_ptr<HiddenNeuron>>& hidden_layer_pointer_temp = *it;
            for(std::shared_ptr<HiddenNeuron>& hidden_n : hidden_layer_pointer_temp) {
                hidden_n->calc_error();
            }
        }
        //Make input neurons calculate error
        for(std::shared_ptr<InputNeuron>& input_n : input_layer) {
            input_n->calc_error();
        }
    }

    void NeuroNet::evaluate_and_propagate_error_calculation(const TrainingSet& set) {
        const InputData& input_data = set.first;
        const OutputData& output_data = set.second;
        evaluate(input_data);
        propagate_error_calculation(output_data);
    }

    void NeuroNet::propagate_weight_change() {
        //Make input neurons change weights
        for(std::shared_ptr<InputNeuron>& input_n : input_layer) {
            input_n->calc_and_add_delta_weight();
        }
        //Make hidden neurons change weights
        for(std::vector<std::shared_ptr<HiddenNeuron>>& hidden_n_layer : hidden_layer) {
            for(std::shared_ptr<HiddenNeuron>& hidden_n : hidden_n_layer) {
                hidden_n->calc_and_add_delta_weight();
            }
        }
        //Make ouput neurons change weights
        for(std::shared_ptr<OutputNeuron>& output_n : output_layer) {
            output_n->calc_and_add_delta_weight();
        }
    }

    void NeuroNet::train_net() {
        for(const TrainingSet& set : training_data) {
            evaluate_and_propagate_error_calculation(set);
            propagate_weight_change();
        }
    }

    void NeuroNet::train_net_convergence() {

        float min_error = __FLT_MAX__;
        std::cout << calculate_mean_squared_error() << std::endl;
        std::cout << "---Training start---" << std::endl;
        float current_error = __FLT_MAX__;
        do{
            //Train network
            train_net();
            if(current_error < min_error) {
                min_error = current_error;
                std::cout << "Increased precision, error=" << min_error << std::endl;
            }
            //Calculate error
            current_error = calculate_mean_squared_error();
        } while(current_error > MIN_NET_PRECISION); //Convergence end condition
        std::cout << "---Training end---" << std::endl;
    }

    void NeuroNet::export_graph_to_file(std::string path) {
        std::ofstream of(path, std::ofstream::out);
        if(!of.is_open()) {
            return;
        }
        //Add all edges to the graph
        for(std::vector<std::shared_ptr<HiddenNeuron>>& hidden_n_layer : hidden_layer) {
            for(std::shared_ptr<HiddenNeuron>& hidden_n : hidden_n_layer) {
                for(std::shared_ptr<NeuronEdge>& edge : hidden_n->getIncomingEdges()) {
                    of << edge->getFrom()->getID() << "-(" << edge->getWeight() << ")-" << edge->getTo()->getID() << "\n";
                }
            }
        }
        for(std::shared_ptr<OutputNeuron>& output_n : output_layer) {
            for(std::shared_ptr<NeuronEdge>& edge : output_n->getIncomingEdges()) {
                of << edge->getFrom()->getID() << "-(" << edge->getWeight() << ")-" << edge->getTo()->getID() << "\n";
            }
        }
        of.close();
        std::cout << "Graph saved to " << path << std::endl;
        std::getchar();
    }

    void NeuroNet::setMinNetPrecision(float precision) {
        MIN_NET_PRECISION = precision;
    }
    float NeuroNet::getMinNetPrecision() const {
        return MIN_NET_PRECISION;
    }

}