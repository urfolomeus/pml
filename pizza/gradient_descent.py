import numpy as np


def predict(X, w, b):
    """
    X is a NumPy array of x coords
    w is a number representing the "weight" or slope of the line
    b is a number representing the "bias" or y-intercept of the line

    Try to predict the value of the y coord based on the value of the x coord
    and the weight of the line.
    """
    return X * w + b


def loss(X, Y, w, b):
    """
    X is a NumPy array of x coords
    Y is a NumPy array of y coords
    w is a number representing the "weight" or slope of the line
    b is a number representing the "bias" or y-intercept of the line

    We square the error so that:
        - we can be sure that it will be a positive number
        - we ensure that the more errors there are, the higher the value will be
    which means that lines with more errors stand out.

    This is known as the `mean squared error`
    """
    return np.average((predict(X, w, b) - Y) ** 2)


def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)


def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=100000, lr=0.001)
print("\nw=%.10f, b=%.10f" % (w, b))
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
