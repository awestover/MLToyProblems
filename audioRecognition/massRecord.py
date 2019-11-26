"""
This program will help you record large ammounts of training and testing data
for machine learning purposes with sound files
"""

import random
import os
import re
import sys

labels = []

labelsFile = "data/possible_labels.txt"
with open(labelsFile) as lf:
	for r in lf:
		labels.append(r.strip())

print(labels)
print("Let's record")
print("Quit the program whenever to stop")



if len(sys.argv) < 2:
	print("input folder to output files to")
	sys.exit(-1)

dataFolder = sys.argv[1]


biggestNumber = -1
for f in os.listdir(dataFolder):
	curNums = re.findall(r'\d+', f)
	print(curNums)
	if len(curNums) == 2:
		curNum = int(curNums[1])
		if curNum > biggestNumber:
			biggestNumber = curNum


def makeFileName(label, number, dataFolder):
	return os.path.join(dataFolder, label + "_" + str(number) + ".wav")

print("We have " + str(biggestNumber) + " sounds so far")

while True:
	biggestNumber += 1
	nextInput = random.choice(labels)
	print("Next you will record for " + nextInput)
	if input("Would you like to quit?\t") == "yes":
		break
	os.system("python3 singleRecord.py " + makeFileName(nextInput, biggestNumber, dataFolder))
	
	