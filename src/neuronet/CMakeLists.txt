add_library(${DEFAULT_NEURON} STATIC Neuron.cpp)
target_include_directories(${DEFAULT_NEURON} PUBLIC Neuron.h)

add_library(${INPUT_NEURON} STATIC InputNeuron.cpp)
target_include_directories(${INPUT_NEURON} PUBLIC InputNeuron.h)

add_library(${HIDDEN_NEURON} STATIC HiddenNeuron.cpp)
target_include_directories(${INPUT_NEURON} PUBLIC HiddenNeuron.h)

add_library(${OUTPUT_NEURON} STATIC OutputNeuron.cpp)
target_include_directories(${INPUT_NEURON} PUBLIC OutputNeuron.h)

add_library(${NEURON_EDGE} STATIC NeuronEdge.cpp)
target_include_directories(${NEURON_EDGE} PUBLIC NeuronEdge.h)

add_library(${NEURO_NET} STATIC NeuroNet.cpp)
target_include_directories(${NEURO_NET} PUBLIC NeuroNet.h)

add_subdirectory("exceptions")