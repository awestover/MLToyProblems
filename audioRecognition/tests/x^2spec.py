import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

data = np.linspace(0, 10, 100)

xx = np.sin(data)

fs = 10

plt.plot(data, xx)
plt.pause(2)
plt.cla()
print(fs)
print(xx)

f, t, Sxx = signal.spectrogram(xx, fs)

print(f)
print(t)
print(Sxx)

plt.pcolormesh(t, f, Sxx)
plt.pause(2)


