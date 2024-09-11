# XOR Gate training data
inputs_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_xor = np.array([0, 1, 1, 0])

# Repeat training for XOR gate
weights_xor = np.array([10, 0.2, -0.75])
weights_xor, epochs_xor, errors_xor = perceptron_training(inputs_xor, outputs_xor, weights_xor, learning_rate)

print(f"Final weights for XOR gate: {weights_xor}")
print(f"Number of epochs taken to converge for XOR gate: {epochs_xor}")
