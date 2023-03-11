#include "Neuron.h"

namespace neuronet {

    Neuron::Neuron() : id{ID}{
        //Increase static ID
        ID++;
        //Fuction definitions
        weighted_sum_function = [&] (const std::vector<std::shared_ptr<NeuronEdge>>& edges) -> float {
            float new_value_in = 0;
            for(const std::shared_ptr<NeuronEdge>& edge : edges) {
                new_value_in += ( edge->getFrom()->getValueOut() * edge->getWeight() );
                //std::cout << "From value out " << edge->getFrom()->getValueOut() << std::endl;
                //std::cout << "Edge weight " <<  edge->getWeight() << std::endl;
            }
            //std::cout << "Weighted sum result " << new_value_in << std::endl;
            return new_value_in;
        };
        activation_function_derivative = [&] (float value) {
            return activation_function(value) * (1 - activation_function(value));
        };
        weighted_sum_over_errors_function = [&] (const std::vector<std::shared_ptr<NeuronEdge>>& edges) -> float {
            float new_value_in = 0;
            for(const std::shared_ptr<NeuronEdge>& edge : edges) {
                new_value_in += ( edge->getTo()->getError() * edge->getWeight() );
            }
            return new_value_in;
        };
        activation_function = [] (float weighted_sum) {
            return 1.0 / (1.0 + exp( (-1) * weighted_sum));
        };
    };

    float Neuron::getValueIn() const {
        return value_in; 
    }
    void Neuron::setValueIn(float value) {
        value_in = value;
    }
    void Neuron::addToValueIn(float value) {
        //std::cout << "Value Before: " << value_;
        value_in += value;
        //std::cout << " Value After: " << value_ << std::endl;
    }
    float Neuron::getValueOut() const {
        return value_out; 
    }
    void Neuron::setValueOut(float value) {
        value_out = value;
    }
    void Neuron::addToValueOut(float value) {
        //std::cout << "Value Before: " << value_;
        value_out += value;
        //std::cout << " Value After: " << value_ << std::endl;
    }
    void Neuron::setError(float error_) {
        error = error_;
        //std::cout << "Set error at neuron " << getID() << "  to " << error_ << std::endl;
    }
    float Neuron::getError() const {
        return error;
    }
    int Neuron::getID() const {
        return id;
    }
    float Neuron::getBias() const {
        return *layer_bias;
    }
    void Neuron::setBias(std::shared_ptr<float> bias) {
        layer_bias = bias;
    }
    void Neuron::addIncomingEdge(std::shared_ptr<NeuronEdge> edge) {
        incoming_edges.push_back(edge);
    }
    void Neuron::addOutgoingEdge(std::shared_ptr<NeuronEdge> edge) {
        outgoing_edges.push_back(edge);
    }
    std::vector<std::shared_ptr<NeuronEdge>>& Neuron::getIncomingEdges() {
        return incoming_edges;
    }
    std::vector<std::shared_ptr<NeuronEdge>>& Neuron::getOutgoingEdges() {
        return outgoing_edges;
    }
}