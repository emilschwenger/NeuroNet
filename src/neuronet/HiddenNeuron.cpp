#include "HiddenNeuron.h"

namespace neuronet {
    std::shared_ptr<HiddenNeuron> HiddenNeuron::getShared() {
        return shared_from_this();
    }
    void HiddenNeuron::fire() {
        for(std::shared_ptr<NeuronEdge>& edge : getOutgoingEdges()) {
            float function_evaluation = getValueFunction()(this->getValue(),FUNCTION_TYPE::DEFAULT_FUNCTION);
            float adding_value = function_evaluation * edge->getWeight();
            //std::cout << "Called Hidden Neuron Fire added -> " << adding_value << std::endl;
            edge->getTo()->addToValue(adding_value);
        }
    }
}