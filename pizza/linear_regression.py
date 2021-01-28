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


def train(X, Y, iterations, lr):
    """
    Tries to find an optimal value for w by stepping it by lr in either direction.
    If it doesn't manage to converge before the given number of iterations, it
    throws an exception.
    """
    w = b = 0

    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b

    raise Exception("Couldn't converge within %d iterations" % iterations)


# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

# predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
