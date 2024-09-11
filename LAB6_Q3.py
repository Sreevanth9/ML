# Function to handle different activation functions
def perceptron_training_with_activation(inputs, outputs, weights, learning_rate, activation_func, max_epochs=1000,
                                        error_threshold=0.002):
    epochs = 0
    errors = []

    while epochs < max_epochs:
        total_error = 0
        for i in range(len(inputs)):
            input_with_bias = np.insert(inputs[i], 0, 1)  # Adding bias input (1)
            summation = summation_unit(input_with_bias, weights)
            prediction = activation_func(summation)
            error = comparator_unit(prediction, outputs[i])
            total_error += error ** 2

            # Weight update rule
            weights += learning_rate * error * input_with_bias

        errors.append(total_error)
        epochs += 1

        if total_error <= error_threshold:
            break

    return weights, epochs, errors


# Comparing different activation functions
activation_functions = [step_function, bipolar_step_function, sigmoid_function, relu_function]
for func in activation_functions:
    weights, epochs, errors = perceptron_training_with_activation(inputs, outputs, weights, learning_rate, func)
    print(f"Activation Function: {func.__name__}, Epochs: {epochs}")
