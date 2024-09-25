# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # Data from your table
# data = np.array([
#     [1.0, -10.332609, -12.959778],
#     [-461.02347, 1.0, -3.3609893],
#     [-1152.1288, -7.3632035, 1.0]
# ])

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Labels for the models
# models = ['T0', 'T1', 'T2']

# # Scatter plot: T0, T1, and T2 on the x, y, z axes
# ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=['r', 'g', 'b'], s=100, label=models)

# # Setting axis labels
# ax.set_xlabel('Likelihood on T0')
# ax.set_ylabel('Likelihood on T1')
# ax.set_zlabel('Likelihood on T2')

# # Adding a legend
# for i in range(len(models)):
#     ax.text(data[i, 0], data[i, 1], data[i, 2], models[i], size=10, zorder=1, color='k')

# # Show plot
# plt.show()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def plot(df):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: T0, T1, and T2 on the x, y, z axes (normalized)
    ax.scatter(df['T0'], df['T1'], df['T2'], c=['r', 'g', 'b'], s=100)

    # Setting axis labels
    ax.set_xlabel('Normalized Likelihood on T0')
    ax.set_ylabel('Normalized Likelihood on T1')
    ax.set_zlabel('Normalized Likelihood on T2')

    # Adding a legend with labels for each model
    for i in range(len(df)):
        ax.text(df['T0'][i], df['T1'][i], df['T2'][i], df.index[i], size=10, zorder=1, color='k')

    # Show plot
    plt.show()
