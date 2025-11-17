import numpy as np
import matplotlib.pyplot as plt
from hopfield import Hopfield
# wybierz ktory obraz zaszumaiamy i odtwarzamy
indx = 0

vec = np.loadtxt("data/animals-14x9.csv", delimiter=",")
# tricky w niektrórych zbiorach
height = 9
width = 14

net = Hopfield()
net.train(vec, height=height, width=width)

img = vec[indx].reshape(height, width)

plt.imshow(img, cmap="gray")
plt.show()

nums_neurons = height * width

pattern = vec[indx].copy()
noisy = pattern.copy()
noise_level = 0.2
idx = np.random.choice(nums_neurons, int(nums_neurons * noise_level), replace=False)
noisy[idx] *= -1

plt.imshow(noisy.reshape(height, width), cmap="gray")
plt.show()

recovered = net.run_once(noisy)

plt.imshow(recovered.reshape(height, width), cmap="gray")
plt.show()
