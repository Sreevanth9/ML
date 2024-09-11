learning_rates = [0.1 * i for i in range(1, 11)]
for lr in learning_rates:
    weights = np.array([10, 0.2, -0.75])
    weights, epochs, errors = perceptron_training(inputs, outputs, weights, lr)
    print(f"Learning Rate: {lr}, Epochs: {epochs}")
