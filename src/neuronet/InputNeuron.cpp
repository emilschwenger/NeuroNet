#include "InputNeuron.h"

namespace neuronet {
    void InputNeuron::addIncomingEdge(std::shared_ptr<NeuronEdge> edge) {
        throw std::runtime_error("InputNeuron can not have incoming edges");
    }
    std::vector<std::shared_ptr<NeuronEdge>>& InputNeuron::getIncomingEdges() {
        throw std::runtime_error("InputNeuron can not have incoming edges");
    }
    std::shared_ptr<InputNeuron> InputNeuron::getShared() {
        return shared_from_this();
    }
    void InputNeuron::fire() {
        for(std::shared_ptr<NeuronEdge>& edge : getOutgoingEdges()) {
            //NOTHING
        }
    }
    void InputNeuron::calc_error() {
        //NOTHING
    }
    void InputNeuron::calc_and_add_delta_weight() {
        //NOTHING
    }
}