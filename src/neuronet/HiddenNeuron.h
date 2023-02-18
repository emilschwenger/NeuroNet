#pragma once
//Standard Library
#include <vector>
#include <memory>
#include <stdexcept>
//User defined
#include "Neuron.h"
#include "NeuronEdge.h"

//TODO: Implement copy and move constructor/assignment

namespace neuronet {

    struct HiddenNeuron : public Neuron , public std::enable_shared_from_this<HiddenNeuron> {
    public:
        explicit HiddenNeuron() = default;
        //Shared pointer generator
        std::shared_ptr<HiddenNeuron> getShared();
        void fire() override;
    };
}