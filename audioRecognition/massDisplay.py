# show all of the audio files 

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd
import numpy as np
import sys
import os


if len(sys.argv) < 2:
    print("Plays wave files. Input folder for wave files to be found in.")
    sys.exit(-1)

dataFolder = sys.argv[1]


def pic_and_csv(fileName):
	wf = wavfile.read(fileName)

	# wf[0] is the sample rate
	# wf[1] is the data (intensities of sound)

	plt.cla()
	plt.title(fileName.replace(".wav", ""))
	plt.ylabel('Intensity')
	plt.xlabel('Time')
	plt.plot(range(0, len(wf[1])), wf[1])
	plt.pause(1)

	plt.cla()
	plt.title(fileName.replace(".wav", "_fft"))
	plt.ylabel('Power')
	plt.xlabel('Frequency')
	f = np.fft.fft(wf[1])
	amps = np.multiply(np.conjugate(f), f)
	lamps = np.log(amps)  # they are by nature exponential, log it to see structure
	shift_lamps = np.fft.fftshift(lamps)  # recenter, to 0. fft splits a bell curve, which is not great
	plt.plot( np.arange(len(amps)), shift_lamps)
	plt.pause(1)

	plt.cla()
	plt.title(fileName.replace(".wav", "_spectogram"))

	fs = wf[0]
	x = wf[1]

	plt.specgram(x, Fs=fs)

	plt.ylabel('Frequency')
	plt.xlabel('Time')
	
	plt.pause(1)


for f in os.listdir(dataFolder):
	if ".wav" in f:
		pic_and_csv( os.path.join(dataFolder, f) )


