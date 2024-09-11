# A8: Neural Network with Backpropagation

# Initialize weights for the neural network
weights_hidden = np.random.rand(3, 2)  # Weights for 2 neurons in the hidden layer (including bias)
weights_output = np.random.rand(3)  # Weights for the output layer (including bias)
learning_rate = 0.05

# Sigmoid Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Backpropagation algorithm for a single hidden layer neural network
def backpropagation_and_gate(inputs, outputs, weights_hidden, weights_output, learning_rate, max_epochs=1000, error_threshold=0.002):
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
            error = comparator_unit(final_output, outputs[i])
            total_error += error ** 2

            # Backward pass (Error Backpropagation)
            delta_output = error * sigmoid_derivative(final_output)
            delta_hidden = delta_output * weights_output[1:] * sigmoid_derivative(hidden_layer_output)

            # Update weights
            weights_output += learning_rate * delta_output * hidden_output_with_bias
            weights_hidden += learning_rate * np.outer(input_with_bias, delta_hidden)

        errors.append(total_error)
        epochs += 1

        if total_error <= error_threshold:
            break

    return weights_hidden, weights_output, epochs, errors

# Train the neural network using backpropagation for AND gate
weights_hidden, weights_output, epochs, errors = backpropagation_and_gate(inputs, outputs, weights_hidden, weights_output, learning_rate)

# Print final weights and epochs taken to converge
print(f"Final hidden weights after training: {weights_hidden}")
print(f"Final output weights after training: {weights_output}")
print(f"Number of epochs taken to converge: {epochs}")
