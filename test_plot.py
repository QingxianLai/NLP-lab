import numpy as np
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
a = [np.log(x) for x in xrange(50)]


for i in xrange(40):
    print i
#     time.sleep(0.5)

# plt.figure(figsize=(10,8))
plt.plot(a, label = "a")
plt.xlabel("aafd")
plt.legend(loc =1, prop={'size':18})
plt.savefig("hhah.png")
