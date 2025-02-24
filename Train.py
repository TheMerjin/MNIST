import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\Sreek\Downloads\Auto_typer\train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape



# Process data: assume each row is one sample (label is first column)



def init_params():
    
    W1 = np.random.randn(256, 784) * 0.01
    b1 = np.zeros((256, 1))
    W15 = np.random.randn(10, 256) * 0.01
    b15 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W15, b15, W2, b2


def Relu(x):
    return np.maximum(x, 0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def forward(x, W1, b1,W15, b15, W2, b2):
    
    Z1 = W1.dot(x) + b1  # (10, m)
    A1 = Relu(Z1)
    Z15 = W15.dot(A1) + b15  # (10, m)
    A15 = Relu(Z15)        # (10, m)
    Z2 = W2.dot(A15) + b2  # (10, m)
    A2 = softmax(Z2)      # (10, m)
    return Z1, A1,Z15, A15, Z2, A2


def onehot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T


def derivative_relu(Z):
    return Z > 0


def back_propogation(Z1, A1, Z15, A15, Z2, A2, W15,W2, x, y):
    m = y.size
    one_hot_encoded_y = onehot(y)
    d_Z2 = A2 - one_hot_encoded_y
    d_Z15 = W2.T.dot(d_Z2) * derivative_relu(Z15)           # (10, m)
    d_Z1 = W15.T.dot(d_Z15) * derivative_relu(Z1)
      # (10, m)
    dW_2 = (1/m) * d_Z2.dot(A15.T)
    dW_15 = (1/m)* d_Z2.dot(A1.T)            # (10, m) dot (m, 10) -> (10, 10)
    dW_1 = (1/m) * d_Z1.dot(x.T)              # (10, m) dot (m, 784) -> (10, 784)
    db_2 = (1/m) * np.sum(d_Z2, axis=1, keepdims=True)
    db_15 = (1/m)*  np.sum(d_Z15, axis=1, keepdims=True)
    db_1 = (1/m) * np.sum(d_Z1, axis=1, keepdims=True)
    return dW_1, dW_15, dW_2, db_1, db_15, db_2


def update_weights(W1, b1,W15,b15, W2, b2, dW_1, dW_2, dW_15, db_1, db_15, db_2, alpha):
    W1 -= alpha * dW_1
    b1 -= alpha * db_1
    W15 -= alpha* dW_15
    b15 -= alpha * db_15
    W2 -= alpha * dW_2
    b2 -= alpha * db_2
    return W1, W15, b1, W2, b15 ,b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(iterations, alpha, x, y):
    W1, b1, W15, b15, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1,Z15, A15, Z2, A2 = forward(x, W1, b1, W15, b15, W2, b2)
        dW_1, dW_15, dW_2, db_1, db_15, db_2 = back_propogation(Z1, A1,Z15, A15, Z2, A2, W15, W2, x, y)
        W1, W15, b1, W2, b15, b2= update_weights(W1, b1,W15,b15, W2, b2, dW_1, dW_2, dW_15, db_1, db_15, db_2, alpha)
        if i % 50 == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), y)}")
    return W1, b1,W15, b15, W2, b2


W1, b1, W15, b15, W2, b2 = gradient_descent(500, 0.5, X_train, Y_train)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, _,_, A2 = forward(X, W1, b1, W15, b15, W2, b2)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
   
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)





dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))






























































































