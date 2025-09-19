import numpy as np
import pandas as pd

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

iterations = 5001
alpha = 0.1

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
    print(predictions[:10], Y[:10])
    return np.sum(predictions == Y) / Y.size

if __name__ == "__main__":
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = backward_prop(A1, Z1, W1, A2, Z2, W2, X_train, Y_train)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 1000 == 0:
            print("Iteration:", i)
            train_preds = get_predictions(A2)
            print("Train Accuracy:", get_accuracy(train_preds, Y_train))

            _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
            test_preds = get_predictions(A2_test)
            print("Test Accuracy:", get_accuracy(test_preds, Y_test))