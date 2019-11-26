from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

fileName = sys.argv[1]

wf = wavfile.read(fileName)

# wf[0] is the sample rate
# wf[1] is the data

plt.plot(range(0, len(wf[1])), wf[1])
plt.pause(1)


df = {
	"data": wf[1]
}
df = pd.DataFrame(df)
df.to_csv("test.csv", index=False)