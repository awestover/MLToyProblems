import matplotlib.pyplot as plt
import numpy as np

fs = 20
t = np.arange(0.0, 20.0, 1/fs)
s1 = np.sin(t)
s2 = t*t
s3 = np.sin(10*t)

plt.plot(t, s2)
plt.pause(2)
plt.cla()

plt.specgram(s2, Fs=fs)
plt.pause(2)
plt.cla()

plt.plot(t, s1)
plt.plot(t, s3)
plt.pause(2)
plt.cla()

plt.specgram(5 * s1, Fs=fs)
plt.specgram(s3, Fs=fs)
plt.pause(2)
plt.cla()

