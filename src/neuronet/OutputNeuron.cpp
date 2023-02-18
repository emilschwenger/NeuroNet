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
        std::cout << "Current output neuron value: " << this->getValue() << std::endl;
    }
}