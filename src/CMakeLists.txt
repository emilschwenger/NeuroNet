add_executable(${EXECUTABLE_NAME} run.cpp)

set(EXECUTABLE_NAME neuro_net)

set(DEFAULT_NEURON default_neuron)
target_link_libraries(${EXECUTABLE_NAME} ${DEFAULT_NEURON})

set(INPUT_NEURON input_neuron)
target_link_libraries(${EXECUTABLE_NAME} ${INPUT_NEURON})

set(HIDDEN_NEURON hidden_neuron)
target_link_libraries(${EXECUTABLE_NAME} ${HIDDEN_NEURON})

set(OUTPUT_NEURON output_neuron)
target_link_libraries(${EXECUTABLE_NAME} ${OUTPUT_NEURON})

set(NEURON_EDGE neuron_edge)
target_link_libraries(${EXECUTABLE_NAME} ${NEURON_EDGE})

set(NEURO_NET neuronet)
target_link_libraries(${EXECUTABLE_NAME} ${NEURO_NET})

set(NEURO_NET_EXCEPTIONS neuronet_exceptions)
target_link_libraries(${EXECUTABLE_NAME} ${NEURO_NET_EXCEPTIONS})

add_subdirectory(neuronet)