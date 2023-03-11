#include "HiddenNeuron.h"

namespace neuronet {
    std::shared_ptr<HiddenNeuron> HiddenNeuron::getShared() {
        return shared_from_this();
    }
    void HiddenNeuron::fire() {
        float new_value_in = weighted_sum_function(getIncomingEdges());
        setValueIn(new_value_in + getBias());
        setValueOut(activation_function(new_value_in));
    }
    void HiddenNeuron::calc_error() {
        float new_error = activation_function_derivative(getValueIn()) * weighted_sum_over_errors_function(getOutgoingEdges());
        setError(new_error);
    }
    void HiddenNeuron::calc_and_add_delta_weight() {
        for(std::shared_ptr<NeuronEdge>& edge : getIncomingEdges()) {
            float delta_weight = (-1) * LEARNING_RATE * getError() * edge->getFrom()->getValueOut();
            std::cout << "Add to hidden weight " << delta_weight << std::endl;
            edge->addToWeight(delta_weight);
        }
    }
}