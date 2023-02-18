#pragma once
//Standard Library
#include <vector>
#include <memory>
#include <stdexcept>
//User defined
#include "Neuron.h"
#include "NeuronEdge.h"
#include "HiddenNeuron.h"

//TODO: Implement copy and move constructor/assignment

namespace neuronet {

    struct InputNeuron : public Neuron, public std::enable_shared_from_this<InputNeuron> {
        explicit InputNeuron() = default;
        //Override incoming edge setter for InputNeuron with an exception
        void addIncomingEdge(std::shared_ptr<NeuronEdge> edge) override;
        //Override get icoming edges
        std::vector<std::shared_ptr<NeuronEdge>>& getIncomingEdges() override;
        // Shared pointer generator
        std::shared_ptr<InputNeuron> getShared();
        void fire() override;
    };
}