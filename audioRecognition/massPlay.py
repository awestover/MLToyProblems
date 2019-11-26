# play all the wave files in a directory
import pyaudio
import time
import wave
import sys
import os

CHUNK = 1024

if len(sys.argv) < 2:
    print("Plays wave files. Input folder for wave files to be found in")
    sys.exit(-1)

dataFolder = sys.argv[1]

def playFile(wfile):
	wf = wave.open(wfile, 'rb')

	p = pyaudio.PyAudio()

	def callback(in_data, frame_count, time_info, status):
	    data = wf.readframes(frame_count)
	    return (data, pyaudio.paContinue)

	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	                channels=wf.getnchannels(),
	                rate=wf.getframerate(),
	                output=True,
	                stream_callback=callback)

	stream.start_stream()

	while stream.is_active():
	    time.sleep(0.1)

	stream.stop_stream()
	stream.close()
	wf.close()

	p.terminate()



for f in os.listdir(dataFolder):
	if ".wav" in f:
		print("Playing " + f)
		playFile( os.path.join(dataFolder, f) )
		time.sleep(0.1)
