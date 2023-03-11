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

    struct OutputNeuron : public Neuron , public std::enable_shared_from_this<OutputNeuron> {
    public:
        explicit OutputNeuron() = default;
        //Override outgoing edge setter for OutputNeuron with an exception
        void addOutgoingEdge(std::shared_ptr<NeuronEdge> edge) override;
        //Override get outgoing edges
        std::vector<std::shared_ptr<NeuronEdge>>& getOutgoingEdges() override;
        //Shared pointer generator
        std::shared_ptr<OutputNeuron> getShared();
        void fire() override;
        void calc_error() override;
        void calc_and_add_delta_weight() override;
        //Expected setter/getter
        float getExpected() const;
        void setExpected(float expected);
    private:
        float expected_output;
    };
}