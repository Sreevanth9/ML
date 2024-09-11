import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Modified backpropagation function for neural networks with 2 output nodes
def backpropagation_two_output_nodes(inputs, outputs, weights_hidden, weights_output, learning_rate, max_epochs=1000, error_threshold=0.002):
    epochs = 0
    errors = []

    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(inputs)):
            # Forward pass
            input_with_bias = np.insert(inputs[i], 0, 1)  # Adding bias input (1)
            hidden_layer_input = np.dot(input_with_bias, weights_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)

            hidden_output_with_bias = np.insert(hidden_layer_output, 0, 1)  # Adding bias for hidden to output layer
            final_input = np.dot(hidden_output_with_bias, weights_output)
            final_output = sigmoid(final_input)

            # Calculate error
            error = outputs[i] - final_output
            total_error += np.sum(error ** 2)

            # Backward pass (Error Backpropagation)
            delta_output = error * sigmoid_derivative(final_output)
            delta_hidden = np.dot(weights_output[1:], delta_output) * sigmoid_derivative(hidden_layer_output)

            # Update weights
            weights_output += learning_rate * np.outer(hidden_output_with_bias, delta_output)
            weights_hidden += learning_rate * np.outer(input_with_bias, delta_hidden)

        errors.append(total_error)
        epochs += 1

        if total_error <= error_threshold:
            break

    return weights_hidden, weights_output, epochs, errors

# AND gate training data with two output nodes (one-hot encoded)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs for AND gate
outputs_and_two_nodes = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # Mapping 0 to [1, 0] and 1 to [0, 1]

# Initialize weights (randomly)
np.random.seed(42)  # Set seed for reproducibility
weights_hidden = np.random.rand(3, 3)  # 3 input nodes (including bias), 3 hidden nodes
weights_output_two_nodes = np.random.rand(4, 2)  # 3 hidden nodes + 1 bias, 2 output nodes

# Parameters
learning_rate = 0.5

# Train the neural network with 2 output nodes for AND gate
weights_hidden_two_nodes, weights_output_two_nodes, epochs_two_nodes, errors_two_nodes = backpropagation_two_output_nodes(
    inputs, outputs_and_two_nodes, weights_hidden, weights_output_two_nodes, learning_rate
)

# Print final weights and epochs taken to converge
print(f"Final hidden weights after training with 2 output nodes:\n{weights_hidden_two_nodes}")
print(f"Final output weights after training with 2 output nodes:\n{weights_output_two_nodes}")
print(f"Number of epochs taken to converge with 2 output nodes: {epochs_two_nodes}")

# Test the network with AND inputs
for input_data in inputs:
    input_with_bias = np.insert(input_data, 0, 1)  # Add bias to the input
    hidden_layer_output = sigmoid(np.dot(input_with_bias, weights_hidden_two_nodes))
    hidden_output_with_bias = np.insert(hidden_layer_output, 0, 1)  # Add bias to hidden layer output
    final_output = sigmoid(np.dot(hidden_output_with_bias, weights_output_two_nodes))
    print(f"Input: {input_data} => Output: {final_output}")
