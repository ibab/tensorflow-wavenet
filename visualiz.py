import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys


file_name = sys.argv[1]

data = np.genfromtxt(file_name ,delimiter=",")
scaled_data = np.kron(data, np.ones([1,100]))
plt.imshow(scaled_data)
# plt.show()
plt.imsave(file_name + ".png", scaled_data, format="png")
