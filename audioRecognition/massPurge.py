import os
import sys

if len(sys.argv) < 3:
	print("input folder to purge of wave files and the type of purge to perform. (notTxt,isWav,badClean)")
	sys.exit(-1)

purgeFolder = sys.argv[1]
ptype = sys.argv[2]

def notTxt(fileName):
	return not ".txt" in fileName

def isWav(fileName):
	return ".wav" in fileName

def badClean(fileName):
	return "clean" in fileName

for f in os.listdir(purgeFolder):
	rm = False
	if ptype == "notTxt":
		if notTxt(f):
			rm = True
	elif ptype == "isWav":
		if isWav(f):
			rm = True
	elif ptype == "badClean":
		if badClean(f):
			rm = True

	if rm:
		os.remove( os.path.join(purgeFolder, f) )