# NeuroNet
A neural network trained with backpropagation.

### Further information
This neural network uses the sigmoid function as the activation function and the mean square error as the error function.

## Functionality

### Create a new neuronal network
```
NeuroNet network {input_layer,hidden_layers,output_layer,training_data};
```
### Mesh the neuronal network
```
network.mesh();
```
### Training
#### Give a network precision
```
network.setMinNetPrecision(0.000005);
```
#### Train untill convergence
```
network.train_net_convergence();
```
### Evaluate network with input data
```
OutputData output_vector = network.evaluate_return(input_data);
```

### Export the network and display it graphically
> https://graphonline.ru/en/create_graph_by_edge_list
```
network.export_graph_to_file("path.txt");
```
