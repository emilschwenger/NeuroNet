#include "NeuronEdge.h"

namespace neuronet {

    NeuronEdge::NeuronEdge(std::shared_ptr<Neuron> from, std::shared_ptr<Neuron> to) : from_{from}, to_{to} {}
    float NeuronEdge::getWeight() const {
        return weight_; 
    }
    void NeuronEdge::setWeight(float weight) {
        weight_ = weight; 
    }
    void NeuronEdge::addToWeight(float weight_delta) {
        weight_ += weight_delta;
    }
    std::shared_ptr<NeuronEdge> NeuronEdge::getShared() {
        return shared_from_this();
    }
    std::shared_ptr<Neuron> NeuronEdge::getFrom() {
        return from_;
    }
    std::shared_ptr<Neuron> NeuronEdge::getTo() {
        return to_;
    }
}