#pragma once
//Standard library types
#include <functional>
#include <optional>
#include <stdexcept>
#include <exception>
#include <vector>
#include <memory>
#include <iostream>
//User defined types
#include "NeuronEdge.h"

namespace neuronet {

    //Enum to compute function type
    enum class FUNCTION_TYPE {
        DEFAULT_FUNCTION, DERIVATIVE_FUNCTION
    };

    //Type for a value function
    using ValueFunction = std::function<float(float,FUNCTION_TYPE)>;

    static ValueFunction NO_VALUE_FUNCTION = [] (float value, FUNCTION_TYPE type) -> float {
        if(type == FUNCTION_TYPE::DEFAULT_FUNCTION) {
            return value;
        }
        if(type == FUNCTION_TYPE::DERIVATIVE_FUNCTION) {
            return 0;
        }
        return 0;
    };

    //Forward decleration of nueron edge
    struct NeuronEdge;

    struct Neuron {
        constexpr static float INITIAL_NEURON_VALUE = 1.0f;
    public:
    /*
        //delete copy constructor and assignment
        Neuron(const Neuron&) = delete;
        Neuron& operator=(const Neuron&) = delete;
        //delete move constructor and assignment
        Neuron(Neuron&&) noexcept = delete;
        Neuron& operator=(Neuron&&) noexcept = delete;
    */
        //Default constructor
        explicit Neuron(float initial_value = INITIAL_NEURON_VALUE);
        //Virtual deconstructor
        virtual ~Neuron() = default;
        //Neuron value operations
        float getValue() const;
        void setValue(float value);
        void addToValue(float value);
        //Setter for edges
        virtual void addIncomingEdge(std::shared_ptr<NeuronEdge> edge);
        virtual void addOutgoingEdge(std::shared_ptr<NeuronEdge> edge);
        //Getter for edges
        virtual std::vector<std::shared_ptr<NeuronEdge>>& getIncomingEdges();
        virtual std::vector<std::shared_ptr<NeuronEdge>>& getOutgoingEdges();
        //Set/Get the value function
        void setValueFunction(ValueFunction value_function);
        ValueFunction getValueFunction() const;
        /*
         * Fire executes a value function and propagates the fire call
         */
        virtual void fire() = 0;
    private:
        float value_;
        std::optional<ValueFunction> value_function_;
        std::vector<std::shared_ptr<NeuronEdge>> incoming_edges;
        std::vector<std::shared_ptr<NeuronEdge>> outgoing_edges;
    };

}