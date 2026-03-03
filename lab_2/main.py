import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], # Input data for XOR
             [0,1],
             [1,0],
             [1,1]])

y = np.array([[0], # Output labels for XOR
              [1],
              [1],
              [0]])

def sigmoid(x): # Sigmoid activation function
    return 1/(1+np.exp(-x))

def sigmoid_d(x): # Derivative of sigmoid function
    return sigmoid(x)*(1-sigmoid(x))

np.random.seed(0) # Seed for reproducibility

w1 = np.random.rand(2,2) 
b1 = np.zeros((1,2))

w2 = np.random.rand(2,1)
b2 = np.zeros((1,1))

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, w2) + b2
    output = sigmoid(z2)

    # Back propagation
    error = (1/2) * np.mean((y - output)**2) # Mean squared error
    error_d = y - output                     # Derivative of error with respect to output
    d_output = error_d * sigmoid_d(z2)       # Derivative of output with respect to z2
    error_hidden = d_output.dot(w2.T)        # Derivative of error with respect to a1
    d_hidden = error_hidden * sigmoid_d(z1)  # Derivative of hidden layer with respect to z1

    # Update weights and biases
    w2 += a1.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate   

    w1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Create mesh grid
xx, yy = np.meshgrid(np.linspace(-0.5,1.5,200),
                     np.linspace(-0.5,1.5,200))

grid = np.c_[xx.ravel(), yy.ravel()]

# Forward pass for grid
z1 = sigmoid(np.dot(grid, w1) + b1)
z2 = sigmoid(np.dot(z1, w2) + b2)
Z = z2.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z > 0.5, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=100)
plt.title("XOR Decision Boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()

print("Final output after training:")
print(output)

print("Final weights and biases:")
print("Weights between input and hidden layer:\n", w1)
print("Biases for hidden layer:\n", b1)
print("Weights between hidden and output layer:\n", w2)
print("Biases for output layer:\n", b2)