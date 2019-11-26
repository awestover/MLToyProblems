import numpy as np
import cv2
import pdb

cap = cv2.VideoCapture('output.avi')

ct = 0

while(cap.isOpened()):
	ret, frame = cap.read()

	if ret:  # is there still something returned?

		ct += 1

		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
   			print(ct)
   	else:
   		break

print(str(ct) + " frames")

cap.release()
cv2.destroyAllWindows()