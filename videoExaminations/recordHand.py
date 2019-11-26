import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename='output.avi',
	fourcc=fourcc, fps=30.0, frameSize=(640,480))

ct = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    ct += 1

    # set frame values to zero
    frame[:, :, 0] = 0
    frame[:, :, 1] = 0

    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print(ct)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

