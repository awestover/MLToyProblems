import pandas as pd
import numpy as np
import sys
import pdb
import os

if len(sys.argv) < 2:
    print("Plays wave files. Input folder for wave files to be found in")
    sys.exit(-1)

dataFolder = sys.argv[1]


for f in os.listdir(dataFolder):
	if ".csv" in f and "_clean" not in f:
		c_data = pd.read_csv(os.path.join(dataFolder, f))

		std_dev = np.std(c_data["data"], ddof=0)  # ddof determines denominator
		mean = np.mean(c_data["data"])
		c_data["data"] = ( c_data["data"] - mean ) / std_dev
		
		# above is mean normilization or something I guess having 0 magnitude could be good too
		# but they get pretty small, not great for sigmoiding

		# c_data["data"] = c_data["data"] / np.linalg.norm(c_data["data"])
		c_data.to_csv(os.path.join(dataFolder, f.replace(".csv", "_clean.csv")), index=False)
