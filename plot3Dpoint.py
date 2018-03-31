import requests
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def plot_3dpoint(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='BLUE')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

plot_3dpoint(3,4,5)
