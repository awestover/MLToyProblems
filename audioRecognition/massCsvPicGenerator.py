# make pictures and csvs for all the wave files
from scipy.io import wavfile
import matplotlib.pyplot as plt
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
	# wf[1] is the data

	plt.cla()
	plt.title(fileName.replace(".wav", ""))
	plt.ylabel('Intensity')
	plt.xlabel('Time')
	plt.plot(range(0, len(wf[1])), wf[1])
	plt.savefig(fileName.replace(".wav", ".png"))

	plt.cla()
	plt.title(fileName.replace(".wav", "_fft"))
	plt.ylabel('Power')
	plt.xlabel('Frequency')
	f = np.fft.fft(wf[1])
	amps = np.multiply(np.conjugate(f), f)
	lamps = np.log(amps)  # they are by nature exponential, log it to see structure
	shift_lamps = np.fft.fftshift(lamps)  # recenter, to 0. fft splits a bell curve, which is not great
	plt.plot( np.arange(len(amps)), shift_lamps)
	plt.savefig(fileName.replace(".wav", "_f.png"))


	plt.cla()
	plt.title(fileName.replace(".wav", "_spectogram"))
	plt.specgram(wf[1], Fs=wf[0])
	plt.ylabel('Frequency')
	plt.xlabel('Time')
	plt.savefig(fileName.replace(".wav", "_spectogram.png"))

	df = {
		"data": wf[1]
	}
	df = pd.DataFrame(df)
	df.to_csv(fileName.replace(".wav", ".csv"), index=False)



for f in os.listdir(dataFolder):
	if ".wav" in f:
		pic_and_csv( os.path.join(dataFolder, f) )