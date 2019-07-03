import matplotlib.pyplot as plt
import numpy as np

from perfana.datasets import load_cube
from perfana.monte_carlo import returns_path

data = load_cube()[..., :7]
weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
quantile = [0.05, 0.25, 0.5, 0.75, 0.95]
color = np.array([
    (200, 200, 200),
    (143, 143, 143),
    (196, 8, 8),
    (143, 143, 143),
    (200, 200, 200),
]) / 255

paths = returns_path(data, weights, quantile=quantile)
n, t = paths.shape
x = np.arange(t)

fig = plt.figure(figsize=(8, 6))
sp = fig.add_subplot(111)
for i in reversed(range(n)):
    label = f"{int(quantile[i] * 100)}"
    sp.plot(x, paths[i], figure=fig, label=label, color=color[i])

sp.fill_between(x, paths[0], paths[-1], color=color[0])
sp.fill_between(x, paths[1], paths[-2], color=color[1])
sp.set_title("Cumulative returns path")
sp.set_xlabel("Time Period")
sp.set_ylabel("Returns")
sp.grid()
sp.legend(title="Percentile")
fig.show()
