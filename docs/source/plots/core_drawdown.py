import matplotlib.pyplot as plt

from perfana.core import drawdown
from perfana.datasets import load_hist

data = load_hist().iloc[:, :7]
weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]

dd = drawdown(data, weights)
plt.plot(dd.index, dd)
plt.title("Drawdown")
plt.xlabel("Time")
plt.ylabel("Drawdown Depth")
plt.show()
