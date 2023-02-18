#include "Neuron.h"

namespace neuronet {
    Neuron::Neuron(float initial_value) : value_{initial_value} {};
    float Neuron::getValue() const {
        return value_; 
    }
    void Neuron::setValue(float value) {
        value_ = value;
    }
    void Neuron::addToValue(float value) {
        //std::cout << "Value Before: " << value_;
        value_ += value;
        //std::cout << " Value After: " << value_ << std::endl;
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
    void Neuron::setValueFunction(ValueFunction value_function) {
        value_function_ = value_function;
    }
    ValueFunction Neuron::getValueFunction() const {
        return value_function_.value_or(NO_VALUE_FUNCTION);
    }
}