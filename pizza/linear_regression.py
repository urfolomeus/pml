import numpy as np


def predict(X, w):
    """
    X is a NumPy array of x coords
    w is a number representing the "weight" or angle of the line

    Try to predict the value of the y coord based on the value of the x coord
    and the weight of the line.
    """
    return X * w


def loss(X, Y, w):
    """
    X is a NumPy array of x coords
    Y is a NumPy array of y coords
    w is a number representing the "weight" or angle of the line

    We square the error so that:
        - we can be sure that it will be a positive number
        - we ensure that the more errors there are, the higher the value will be
    which means that lines with more errors stand out.

    This is known as the `mean squared error`
    """
    return np.average((predict(X, w) - Y) ** 2)


def train(X, Y, iterations, lr):
    """
    Tries to find an optimal value for w by stepping it by lr in either direction.
    If it doesn't manage to converge before the given number of iterations, it
    throws an exception.
    """
    w = 0

    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w

    raise Exception("Couldn't converge within %d iterations" % iterations)


# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system
w = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f" % w)

# predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))
