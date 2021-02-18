import numpy as np


def predict(X, w):
    """
    X is a NumPy matrice containing the model inputs
    w is a number representing the "weight" or slope of the line
    b is a number representing the "bias" or y-intercept of the line

    Try to predict the value of y based on the values of the X inputs
    and the weight of the line.
    """
    return np.matmul(X, w)


def loss(X, Y, w):
    """
    X is a NumPy matrice containing the model inputs
    Y is a NumPy matrice representing the ground truth
    w is a number representing the "weight" or slope of the line
    b is a number representing the "bias" or y-intercept of the line

    We square the error so that:
        - we can be sure that it will be a positive number
        - we ensure that the more errors there are, the higher the value will be
    which means that lines with more errors stand out.

    This is known as the `mean squared error`
    """
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w


x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=100000, lr=0.001)
# print("\nw=%.10f, b=%.10f" % (w, b))
# print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
