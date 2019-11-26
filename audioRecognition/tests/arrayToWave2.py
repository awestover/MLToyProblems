import numpy as np
import wavio

rate = 22050  # samples per second
T = 3         # sample duration (seconds)
f = 440.0     # sound frequency (Hz)
t = np.linspace(0, T, T*rate, endpoint=False)
x = np.sin(2*np.pi * f * t)
wavio.write("sine24.wav", x, rate, sampwidth=3)


rate = 1024
T = 5
t = np.linspace(0, T, T*rate, endpoint=False)
f1 = 440.0
f2 = 880.0

a1 = 1
a2 = 5

x = np.sin(2*np.pi * f1 * t) + np.cos(2*np.pi * f2 * t)

wavio.write("noise.wav", x, rate, sampwidth=3)