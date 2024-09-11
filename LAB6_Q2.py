# Training data for AND Gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])  # AND Gate outputs
weights = np.array([10, 0.2, -0.75])  # Including bias as weight
learning_rate = 0.05


# Activation function - Step function
def perceptron_training(inputs, outputs, weights, learning_rate, max_epochs=1000, error_threshold=0.002):
    epochs = 0
    errors = []

    while epochs < max_epochs:
        total_error = 0
        for i in range(len(inputs)):
            input_with_bias = np.insert(inputs[i], 0, 1)  # Adding bias input (1)
            summation = summation_unit(input_with_bias, weights)
            prediction = step_function(summation)
            error = comparator_unit(prediction, outputs[i])
            total_error += error ** 2

            # Weight update rule
            weights += learning_rate * error * input_with_bias

        errors.append(total_error)
        epochs += 1

        if total_error <= error_threshold:
            break

    return weights, epochs, errors


# Train the perceptron
weights, epochs, errors = perceptron_training(inputs, outputs, weights, learning_rate)

# Print final weights and epochs taken to converge
print(f"Final weights after training: {weights}")
print(f"Number of epochs taken to converge: {epochs}")
