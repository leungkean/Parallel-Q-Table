import matplotlib.pyplot as plt
import numpy as np

training_time = np.array([8924.662913322449, 10522.630375862122, 6465.515100240707, 5680.357272148132, 3914.1790568828583, 4941.82651758194])
workers = np.array([1, 2, 4, 10, 20, 40])
episodes = np.array([100000, 90000, 60000, 45000, 30000, 20000])

plt.figure()
plt.plot(workers, training_time, 'o')
plt.xlabel('Number of Workers (Processes)')
plt.ylabel('Training Time (s)')
plt.grid()
plt.show()

plt.figure()
plt.plot(workers, episodes, 'o')
plt.xlabel('Number of Workers (Processes)')
plt.ylabel('Number of Episodes')
plt.grid()
plt.show()
