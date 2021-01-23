import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# activate seaborn
sns.set()

# scale axes (0 to 50), set ticks and labels on axes
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)

# load the data from the file
X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)

# plot and show data
plt.plot(X, Y, "bo")
plt.show()
