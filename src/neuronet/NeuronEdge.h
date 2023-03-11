#pragma once
//Standard Library
#include <array>
#include <memory>
#include <stdlib.h>
//User defined
#include "Neuron.h"

namespace neuronet {

    //Forward decleration of neuron
    struct Neuron;

    struct NeuronEdge : public std::enable_shared_from_this<NeuronEdge> {
    public:
        explicit NeuronEdge(std::shared_ptr<Neuron> from, std::shared_ptr<Neuron> to);
        float getWeight() const;
        void setWeight(float weight);
        void addToWeight(float weight_delta);
        //Get pointer from/to
        std::shared_ptr<Neuron> getFrom();
        std::shared_ptr<Neuron> getTo();
        //Shared pointer generator
        std::shared_ptr<NeuronEdge> getShared();
    private:
        std::shared_ptr<Neuron> from_;
        std::shared_ptr<Neuron> to_;
        float weight_ = (float) rand()/RAND_MAX;
    };
}