import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

with open('data-1.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

time = []
pot = []

for i in range(1, len(data)):
    time.append(float(data[i][0]))
    pot.append(float(data[i][1]))

time = np.array(time)
pot = np.array(pot)

plt.plot(time, pot, 'r.', markersize=1,
        linestyle='--', linewidth=0.5)
plt.savefig("potvtime.pdf")
