import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Backpropagation function for XOR gate
def backpropagation_xor(inputs, outputs, weights_hidden, weights_output, learning_rate, epochs=10000):
    # Initialize error tracking
    errors = []
    for epoch in range(epochs):
        # Forward pass
        hidden_input = np.dot(inputs, weights_hidden)
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, weights_output)
        final_output = sigmoid(final_input)

        # Calculate error (difference between expected and predicted outputs)
        error = outputs - final_output
        errors.append(np.mean(np.abs(error)))

        # Backpropagation
        output_delta = error * sigmoid_derivative(final_output)
        hidden_error = output_delta.dot(weights_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

        # Update weights
        weights_output += hidden_output.T.dot(output_delta) * learning_rate
        weights_hidden += inputs.T.dot(hidden_delta) * learning_rate

    return weights_hidden, weights_output, epoch, errors

# XOR gate training data
inputs_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_xor = np.array([[0], [1], [1], [0]])

# Initialize weights (randomly)
np.random.seed(42)  # Set seed for reproducibility
weights_hidden = np.random.uniform(size=(2, 2))  # 2 input nodes, 2 hidden nodes
weights_output = np.random.uniform(size=(2, 1))  # 2 hidden nodes, 1 output node

# Parameters
learning_rate = 0.5

# Train the neural network using backpropagation for XOR gate
weights_hidden_xor, weights_output_xor, epochs_xor, errors_xor = backpropagation_xor(
    inputs_xor, outputs_xor, weights_hidden, weights_output, learning_rate
)

# Print final weights and epochs taken to converge
print(f"Final hidden weights after training for XOR gate:\n{weights_hidden_xor}")
print(f"Final output weights after training for XOR gate:\n{weights_output_xor}")
print(f"Number of epochs taken to converge for XOR gate: {epochs_xor}")

# Test the network with XOR inputs
hidden_output_test = sigmoid(np.dot(inputs_xor, weights_hidden_xor))
final_output_test = sigmoid(np.dot(hidden_output_test, weights_output_xor))
print("\nPredictions after training:")
print(final_output_test)
