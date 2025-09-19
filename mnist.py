import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = pd.read_csv("mnist.csv")

data = np.array(data)
r, c = data.shape
np.random.shuffle(data)

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:c]

data_train = data[1000:r].T
Y_train = data_train[0]
X_train = data_train[1:c]

X_train = X_train / 255.0
X_test = X_test / 255.0


W1 = np.random.rand(10, 784) - 0.5
b1 = np.random.rand(10, 1) - 0.5
W2 = np.random.rand(10, 10) - 0.5
b2 = np.random.rand(10, 1) - 0.5

epochs = 5001
alpha = 0.1

iterations = []
train_acc = []
test_acc = []

fig, ax = plt.subplots()
(train_line,) = ax.plot([], [], label="Train Accuracy", color="blue")
(test_line,) = ax.plot([], [], label="Test Accuracy", color="orange")

ax.set_xlim(0, epochs)
ax.set_ylim(0, 1)
ax.set_xlabel("Iteration")
ax.set_ylabel("Accuracy")
ax.legend(loc="lower right")

def init():
    train_line.set_data([], [])
    test_line.set_data([], [])
    return train_line, test_line

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0))  # stability trick
    return exp_x / np.sum(exp_x, axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def ReLU_deriv(x):
    return x > 0

def backward_prop(A1, Z1, W1, A2, Z2, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_decent(frame):
    global W1, b1, W2, b2
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
    dW1, db1, dW2, db2 = backward_prop(A1, Z1, W1, A2, Z2, W2, X_train, Y_train)
    W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    train_val = get_accuracy(get_predictions(A2), Y_train)
    _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
    test_val = get_accuracy(get_predictions(A2_test), Y_test)
    
    iterations.append(frame)
    train_acc.append(train_val)
    test_acc.append(test_val)

    train_line.set_data(iterations, train_acc)
    test_line.set_data(iterations, test_acc)
    return train_line, test_line


ani = FuncAnimation(fig, gradient_decent, frames=np.arange(1, epochs+1), init_func=init, blit=True, interval=100, repeat=False)
plt.show()
