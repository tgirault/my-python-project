import numpy as np


# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# ReLU and its derivative
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Initialize weights and biases
input_size = 3  # Number of input features
hidden_size = 4  # Number of neurons in the hidden layer
output_size = 1  # Output layer (binary classification)

W1 = np.random.randn(hidden_size, input_size)
b1 = np.random.randn(hidden_size, 1)
W2 = np.random.randn(output_size, hidden_size)
b2 = np.random.randn(output_size, 1)

# Training loop
learning_rate = 0.01
for epoch in range(10000):
    # Forward pass
    X = np.random.randn(input_size, 1)  # Random input example
    y = np.array([[1]])  # Example label

    # Hidden layer
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # Output layer
    Z2 = np.dot(W2, A1) + b2
    y_hat = sigmoid(Z2)

    # Compute loss (Binary Cross-Entropy)
    loss = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    # Backpropagation
    dZ2 = y_hat - y
    dW2 = np.dot(dZ2, A1.T)
    db2 = dZ2

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T)
    db1 = dZ1

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


