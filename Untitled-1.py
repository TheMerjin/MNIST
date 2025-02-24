import numpy as np

# Input data and labels
training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
labels = np.array([[0,1,1,0]]).T  # Labels are transposed

# Number of samples and features
m, n = training_inputs.shape

# Initialize weights and biases
def init_params():
    W1 = np.random.randn(n, 2)  # Weights for the first layer (2 input features, 3 neurons)
    b1 = np.random.randn(4, 2)  # Bias for the first layer (3 neurons)
    W2 = np.random.randn(2, 1)  # Weights for the second layer (3 neurons, 1 output)
    b2 = np.random.randn(1, 1)  # Bias for the output layer (1 output)
    return W1, b1, W2, b2

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)

# Forward propagation
def forward(inputs, W1, b1, W2, b2):
    Z1 = inputs.dot(W1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Backpropagation
def back_propagation(Z1, A1, Z2, A2, W2, training_inputs, labels):
    dZ2 = A2 - labels
    dZ1 = dZ2.dot(W2.T) * derivative_sigmoid(A1)
    dW2 = 1/m * A1.T.dot(dZ2)
    db2 = 1/m * np.sum(dZ2, axis=0, keepdims=True)
    dW1 = 1/m * training_inputs.T.dot(dZ1)
    db1 = 1/m * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

# Update weights and biases
def update_weights(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha):
    W1 -= dW1 * alpha
    b1 -= db1 * alpha
    W2 -= dW2 * alpha
    b2 -= db2 * alpha
    return W1, b1, W2, b2

# Calculate accuracy
def get_accuracy(x, y):
    correct = np.sum(np.abs(x - y) < 0.01)  # Count values within a small tolerance
    accuracy = correct / y.shape[0]
    return accuracy

# Training function
def training(iterations, alpha, x, y):
    W1, b1, W2, b2 = init_params()

    for _ in range(iterations):
        Z1, A1, Z2, A2 = forward(x, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W2, x, y)
        W1, b1, W2, b2 = update_weights(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha)

        if _ % 500 == 0:
            print(f"iteration: {_}")
            print("Accuracy:", get_accuracy(A2, y))
            print("Predictions:", A2)
            print("Labels:", y)

    return W1, b1, W2, b2

# Training the model
W1, b1, W2, b2 = training(1000000, 0.01, training_inputs, labels)  # Train for fewer iterations
test_input = np.array([[1, 1, 1]])  # New input for testing (reshape if necessary)

# Forward pass to get the prediction
_, _, _, test_output = forward(test_input, W1, b1, W2, b2)

# Output the result
print("Prediction for input [1, 1, 1]:", test_output)

