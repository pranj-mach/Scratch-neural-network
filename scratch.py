import numpy as np

# ✅ Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)

# ✅ Loss Function
def compute_loss(y_true, y_pred):
    m = y_true.shape[1]
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

# ✅ Initialize Weights
def init_weights(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    W1 = np.random.randn(hidden_dim, input_dim) * 0.01
    bias1 = np.zeros((hidden_dim, 1))
    W2 = np.random.randn(output_dim, hidden_dim) * 0.01
    bias2 = np.zeros((output_dim, 1))
    return W1, bias1, W2, bias2

# ✅ Forward Propagation
def forward_pass(X, W1, bias1, W2, bias2):
    hidden_input = np.dot(W1, X) + bias1
    hidden_output = relu(hidden_input)

    final_input = np.dot(W2, hidden_output) + bias2
    final_output = sigmoid(final_input)

    return hidden_input, hidden_output, final_input, final_output

# ✅ Backpropagation
def backprop(X, Y, W1, bias1, W2, bias2, lr=0.1):
    m = X.shape[1]

    # Forward Pass
    hidden_input, hidden_output, final_input, final_output = forward_pass(X, W1, bias1, W2, bias2)

    # Compute Loss
    loss = compute_loss(Y, final_output)

    # Compute Gradients
    error_final = final_output - Y
    dW2 = (1 / m) * np.dot(error_final, hidden_output.T)
    db2 = (1 / m) * np.sum(error_final, axis=1, keepdims=True)

    error_hidden = np.dot(W2.T, error_final) * relu_grad(hidden_input)
    dW1 = (1 / m) * np.dot(error_hidden, X.T)
    db1 = (1 / m) * np.sum(error_hidden, axis=1, keepdims=True)

    # Update Weights
    W1 -= lr * dW1
    bias1 -= lr * db1
    W2 -= lr * dW2
    bias2 -= lr * db2

    return W1, bias1, W2, bias2, loss

# ✅ Training Function
def train_network(X, Y, hidden_dim=4, epochs=10000, lr=0.1):
    input_dim = X.shape[0]
    output_dim = Y.shape[0]

    # Initialize Parameters
    W1, bias1, W2, bias2 = init_weights(input_dim, hidden_dim, output_dim)

    for epoch in range(epochs):
        W1, bias1, W2, bias2, loss = backprop(X, Y, W1, bias1, W2, bias2, lr)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W1, bias1, W2, bias2

# ✅ Make Predictions
def predict(X, W1, bias1, W2, bias2):
    _, _, _, output = forward_pass(X, W1, bias1, W2, bias2)
    return (output > 0.5).astype(int)

# ✅ XOR Dataset
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # (2, 4)
Y = np.array([[0, 1, 1, 0]])  # (1, 4)

# Train the Model
W1, bias1, W2, bias2 = train_network(X, Y, hidden_dim=4, epochs=10000, lr=0.1)

# Make Predictions
predictions = predict(X, W1, bias1, W2, bias2)
accuracy = np.mean(predictions == Y) * 100

print("\nFinal Accuracy:", accuracy, "%")
