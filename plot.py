import os
import re
import matplotlib.pyplot as plt  
import numpy as np
from scipy.signal import savgol_filter
from tabulate import tabulate
        
p = re.compile("n = ([0-9]+), thread count: ([0-9]+), time: (.*)ms")

Ys = []
X = []
N = 8000
labels = []

Pes = []

for log in os.listdir('results'):
    if '.txt' not in log:
        continue
    with open(f'results/{log}') as f:
        N = 8000
        x = []
        y = []
        pe = []
        for line in reversed(f.readlines()):
            line = line.strip()
            res = re.search(p, line)
            n = int(res.group(1))
            tc = int(res.group(2))
            tm = float(res.group(3))
            x.append(tc)
            y.append(tm)
            ts = y[0]
            pp = ts / (tm *tc)
            pe.append(pp)
            # print()
        X = x
        pe = savgol_filter(pe, 16, 2)
        pe = [min(1.0, ppp) for ppp in pe]
        pe[0] = 1.0
        Ys.append(y)
        Pes.append(pe)
        labels.append(log.split(".")[0])


plt.plot(x, Pes[0], label=labels[0], color='red') 
plt.plot(x, Pes[1], label=labels[1], color='blue') 
plt.plot(x, Pes[2], label=labels[2], color='green') 


plt.xlabel('num threads') 
plt.ylabel('Parallel efficiency') 

plt.legend()

plt.savefig("results/plot.png")



head = ["num. threads"] + labels
data = [[X[i], Ys[0][i], Ys[1][i], Ys[2][i]] for i in range(len(X))]

print(tabulate(data, headers=head, tablefmt="grid"))