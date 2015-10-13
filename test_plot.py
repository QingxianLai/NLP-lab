import matplotlib.pyplot as plt
import seaborn
import numpy as np

a = [np.log(x) for x in xrange(50)]

plt.figure(figsize=(10,8))
plt.plot(a, label = "a")
plt.xlabel("aafd")
plt.legend(loc =1, prop={'size':18})
plt.savefig("hhah.png")
