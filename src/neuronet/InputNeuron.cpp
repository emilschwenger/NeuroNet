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
            float function_evaluation = getValueFunction()(this->getValue(),FUNCTION_TYPE::DEFAULT_FUNCTION);
            float adding_value = function_evaluation * edge->getWeight();
            //std::cout << "Called Input Neuron Fire added -> " << adding_value << std::endl;
            edge->getTo()->addToValue(adding_value);
        }
    }
}