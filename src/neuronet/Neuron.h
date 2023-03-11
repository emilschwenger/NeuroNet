#pragma once
//Standard library types
#include <functional>
#include <optional>
#include <stdexcept>
#include <exception>
#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
//User defined types
#include "NeuronEdge.h"

namespace neuronet {

    //Forward decleration of nueron edge
    struct NeuronEdge;
    
    //ID's for created neurons
    static int ID = 0;

    struct Neuron {
        constexpr static float LEARNING_RATE = 0.50f;
    public:
        //Default constructor
        explicit Neuron();
        //Virtual deconstructor
        virtual ~Neuron() = default;
        //Neuron value in operations
        float getValueIn() const;
        void setValueIn(float value);
        void addToValueIn(float value);
        //Neuron value out operations
        float getValueOut() const;
        void setValueOut(float value);
        void addToValueOut(float value);
        //Bias operations
        float getBias() const;
        void setBias(std::shared_ptr<float> bias);
        //Error operations
        void setError(float error);
        float getError() const;
        //UUID operations
        int getID() const;
        //Setter for edges
        virtual void addIncomingEdge(std::shared_ptr<NeuronEdge> edge);
        virtual void addOutgoingEdge(std::shared_ptr<NeuronEdge> edge);
        //Getter for edges
        virtual std::vector<std::shared_ptr<NeuronEdge>>& getIncomingEdges();
        virtual std::vector<std::shared_ptr<NeuronEdge>>& getOutgoingEdges();
        /*
         * Fire executes a value function and propagates the fire call
         */
        virtual void fire() = 0;
        /*
         * Calc error calculates the error for the specific neuron
         */
        virtual void calc_error() = 0;
        /*
         * Calc delta weight
         */
        virtual void calc_and_add_delta_weight() = 0;
    private:
        //error
        float error;
        //values
        float value_in;
        float value_out;
        //Bias
        std::shared_ptr<float> layer_bias;
        //unique UUID
        int id;
        //cached cost derived on activation value c derived a_k^L
        float derivative_total_cost_on_activation;
    protected:
        //Edges
        std::vector<std::shared_ptr<NeuronEdge>> incoming_edges;
        std::vector<std::shared_ptr<NeuronEdge>> outgoing_edges;
        //forwardpropagation functions
        std::function<float(float)> activation_function;
        /*std::function<float(float)> activation_function = [] (float weighted_sum) {
            return 1.0 / (1 + pow(M_E, (-1) * weighted_sum));
        };*/
        //backwardpropagation functions
        std::function<float(const std::vector<std::shared_ptr<NeuronEdge>>&)> weighted_sum_function;
        /*std::function<float(const std::vector<std::shared_ptr<NeuronEdge>>&)> weighted_sum_function = [&] (const std::vector<std::shared_ptr<NeuronEdge>>& edges) -> float {
            float new_value_in = 0;
            for(const std::shared_ptr<NeuronEdge>& edge : edges) {
                new_value_in += ( edge->getFrom()->getValueOut() * edge->getWeight() );
            }
            return new_value_in;
        };*/
        std::function<float(float)> activation_function_derivative;
        /*std::function<float(float)> activation_function_derivative = [&] (float value) {
            return activation_function(value) * (1 - activation_function(value));
        };*/
        std::function<float(const std::vector<std::shared_ptr<NeuronEdge>>&)> weighted_sum_over_errors_function;
        /*std::function<float(const std::vector<std::shared_ptr<NeuronEdge>>&)> weighted_sum_over_errors_function = [&] (const std::vector<std::shared_ptr<NeuronEdge>>& edges) -> float {
            float new_value_in = 0;
            for(const std::shared_ptr<NeuronEdge>& edge : edges) {
                new_value_in += ( edge->getTo()->getError() * edge->getWeight() );
            }
            return new_value_in;
        };*/
    };

}