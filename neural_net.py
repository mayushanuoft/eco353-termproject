import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Load and preprocess data
data_path = '/Users/mayushanmayurathan/Documents/School/ECO353/term project/corporate_credit_risk.csv'  # Update this path
data = pd.read_csv(data_path)
X = data.drop(['Num', 'Default'], axis=1).values  # Assuming 'Num' is an identifier
y = data['Default'].values  # Assuming 'Default Rate' is the continuous target for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural network helper functions
def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def relu(Z):
    return np.maximum(0, Z)

def compute_mse_cost(AL, Y):
    m = Y.size
    cost = (1/m) * np.sum((AL - Y)**2)
    return cost

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# Forward propagation for regression (with linear activation on the output layer)
def forward_propagation_regression(X, parameters):
    caches = []
    A = X.T
    L = len(parameters) // 2
    for l in range(1, L+1):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        if l == L:  # If it's the last layer, use linear activation
            A = Z
        else:
            A = relu(Z)  # For all other layers, use ReLU activation
        caches.append((A_prev, W, b, Z))
    return A, caches

# Backward propagation for regression
def backward_propagation_regression(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # Y should be reshaped to (1, m) before calling this function
    dAL = 2 * (AL - Y)  # Derivative of MSE cost
    for l in reversed(range(L)):
        current_cache = caches[l]
        A_prev, W, b, Z = current_cache
        if l == L-1:
            dZ = dAL  # Derivative of linear activation is 1
        else:
            dZ = relu_backward(dAL, Z)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db
        dAL = dA_prev
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # Number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters


# Training the regression model
def neural_network_model_regression(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = forward_propagation_regression(X, parameters)
        cost = compute_mse_cost(AL, Y)
        grads = backward_propagation_regression(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("MSE cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    return parameters

# Training the regression model
layers_dims = [X_train.shape[1], 64, 32, 1]  # Example layer dimensions
learning_rate = 0.0075
num_iterations = 3000

# Before training, ensure y_train is shaped correctly
y_train = y_train.reshape(1, -1)

# Train the model
parameters = neural_network_model_regression(X_train, y_train, layers_dims, learning_rate, num_iterations)

# Save the trained parameters for later use
np.savez("trained_regression_model_parameters.npz", **parameters)

# Function to make predictions using the trained model
def predict_regression(X, parameters):
    AL, _ = forward_propagation_regression(X, parameters)
    return AL.T  # Transpose to get predictions in proper shape

# Predict default rates using the test set
predicted_default_rates = predict_regression(X_test, parameters)
# print(predicted_default_rates)
predicted_default_rates_flat = [rate[0] for rate in predicted_default_rates]

# # Plotting the histogram
# plt.figure(figsize=(10, 6))
# plt.hist(predicted_default_rates_flat, bins=30, alpha=0.75, color='blue')
# plt.title('Histogram of Predicted Default Rates')
# plt.xlabel('Predicted Default Rate')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
losses = []  # Initialize a list to store the loss values

for i in range(num_iterations):
    # Perform forward propagation
    AL, caches = forward_propagation_regression(X_train, parameters)
    
    # Compute cost
    cost = compute_mse_cost(AL, y_train)
    losses.append(cost)  # Save the cost for plotting
    
    # Backward propagation
    grads = backward_propagation_regression(AL, y_train, caches)
    
    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)
    
    # Optionally print the cost every 100 iterations
    if i % 100 == 0:
        print("Cost after iteration %i: %f" % (i, cost))

# Plotting the loss curve
plt.plot(losses)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

def print_network_architecture(parameters):
    print("Neural Network Architecture:")
    L = len(parameters) // 2  # Number of layers in the neural network
    for l in range(1, L + 1):
        W = parameters['W' + str(l)].shape
        print(f"Layer {l}:")
        print(f"  - Weights shape: {W}")
        print(f"  - Activation: {'Linear' if l == L else 'ReLU'}")

# Initialize parameters
parameters = initialize_parameters(layer_dims)

# Print network architecture
print_network_architecture(parameters)

# Train the model
parameters = neural_network_model_regression(X_train, y_train, layers_dims, learning_rate, num_iterations)



