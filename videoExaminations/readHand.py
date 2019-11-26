import matplotlib.pyplot as plt
import numpy as np
import cv2
import pdb

# read in the input
cap = cv2.VideoCapture('hand.avi')

# frame count
ct = 0

r_vals = []
r_diffs = []  # r_diffs[i] = r_vals[i] - r_vals[i-1]

itts = []

vid_ct = 10
ct = 0

# read all the frames
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        if (ct+1) % vid_ct == 0:
            print("plotting")
            plt.cla()
            plt.ylim(-10, 10)
            plt.plot(itts[0:ct-1], r_diffs, c='r')
            plt.pause(0.1)

        # set frame values to zero
        frame[:, :, 0] = 0
        frame[:, :, 1] = 0

        n_fac = frame.shape[0] * frame.shape[1]

        r_vals.append(np.sum(frame[:,:,2])/n_fac)
        if ct > 0:
            r_diffs.append(r_vals[ct] - r_vals[ct-1])
        itts.append(ct)
        ct += 1

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print(str(ct) + " frames")

plt.cla()
plt.ylim(-1, 1)
plt.plot(itts[0:ct-1], r_diffs, c='r')
plt.pause(1)
plt.savefig("r_graph.png")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

