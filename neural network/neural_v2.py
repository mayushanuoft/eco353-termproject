import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load and preprocess data
data_path = '/Users/mayushanmayurathan/Documents/School/ECO353/term project/corporate_credit_risk.csv'  # Update this path with your actual data path
data = pd.read_csv(data_path)
X = data.drop(['Num', 'Default'], axis=1).values  # Assuming 'Num' is an identifier and 'Default' is the target
y = data['Default'].values  # Assuming 'Default' is the continuous target for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural network helper functions
def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)  # Number of layers in the neural network
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

def forward_propagation_regression(X, parameters):
    caches = []
    A = X.T
    L = len(parameters) // 2  # Number of layers
    for l in range(1, L+1):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        if l == L:  # Linear activation in the last layer
            A = Z
        else:
            A = relu(Z)
        caches.append((A_prev, W, b, Z))
    return A, caches

def backward_propagation_regression(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # ensure Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = relu_backward(grads["dA" + str(l + 2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters

# Define the network architecture
input_size = X_train.shape[1]  # This should match the number of features in your input dataset
output_size = 1  # Adjust this based on your output size (e.g., regression output)
layer_dims = [input_size, 64, 32, output_size]  # Example layer dimensions

# Initialize network parameters
parameters = initialize_parameters(layer_dims)

# Function to print network architecture
def print_network_architecture(parameters):
    print("Neural Network Architecture:")
    L = len(parameters) // 2  # Number of layers in the neural network
    for l in range(1, L + 1):
        W = parameters['W' + str(l)].shape
        print(f"Layer {l}:")
        print(f"  - Weights shape: {W}")
        print(f"  - Activation: {'Linear' if l == L else 'ReLU'}")

# Print the architecture
print_network_architecture(parameters)
