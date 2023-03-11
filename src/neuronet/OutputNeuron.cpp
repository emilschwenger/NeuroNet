#include "OutputNeuron.h"

namespace neuronet {
    void OutputNeuron::addOutgoingEdge(std::shared_ptr<NeuronEdge> edge) {
        throw std::runtime_error("OuputNeuron can not have outgoing edges");
    }
    std::vector<std::shared_ptr<NeuronEdge>>& OutputNeuron::getOutgoingEdges() {
        throw std::runtime_error("OuputNeuron can not have outgoing edges");
    }
    std::shared_ptr<OutputNeuron> OutputNeuron::getShared() {
        return shared_from_this();
    }
    void OutputNeuron::fire() {
        float new_value_in = weighted_sum_function(getIncomingEdges());
        setValueIn(new_value_in + getBias());
        setValueOut(activation_function(new_value_in));
        std::cout << "Current output neuron value: " << this->getValueOut() << std::endl;
    }
    float OutputNeuron::getExpected() const {
        return expected_output;
    }
    void OutputNeuron::setExpected(float expected) {
        expected_output = expected;
        std::cout << "Set expected at neuron " << getID() << "  to " << expected << std::endl;
    }
    void OutputNeuron::calc_error() {
        float new_error = (getValueOut() - getExpected()) * activation_function_derivative(getValueIn());
        setError(new_error);
    }
    void OutputNeuron::calc_and_add_delta_weight() {
        for(std::shared_ptr<NeuronEdge>& edge : getIncomingEdges()) {
            float delta_weight = (-1) * LEARNING_RATE * getError() * edge->getFrom()->getValueOut();
            std::cout << "Add to output weight " << delta_weight << std::endl;
            edge->addToWeight(delta_weight);
        }
    }
}