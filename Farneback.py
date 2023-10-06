import cv2 as cv
import numpy as np
import time


def fanerback_method(video):
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        print("ERROR! Unable to open video or camera\n")

    ret, first_frame = cap.read()  # ret is a boolean (everything is correct) and first_frame saves the first frame.

    # Resize image
    new_size = (320, 240)  # width, height
    first_frame = cv.resize(first_frame, new_size)

    # Converts frame to grayscale because we only need the luminance channel for detecting edges (less
    # computationally expensive)
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Frame empty!\n")
            break

        new_size = (320, 240)
        frame = cv.resize(frame, new_size)

        # Opens a new window and displays the input frame
        cv.imshow("input", frame)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # To obtain the computational cost we will save the time before and after the method
        start = time.time_ns()

        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        end = time.time_ns()
        cost_ns = end - start
        cost = cost_ns / 1e9 if cost_ns >= 0 else 0

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Dense optical flow representation:
        # The magnitude and direction of optical flow is encoded as a 2-channel array of flow vectors. It then visualizes
        # the angle (direction) of flow by hue and the distance (magnitude) of flow by value of HSV color representation.
        # The strength of HSV is always set to a maximum of 255 for optimal visibility. More details about HSV color space:
        # https://en.wikipedia.org/wiki/HSL_and_HSV
        # Sets image hue according to the optical flow direction
        mask[..., 0] = (angle * 180 / np.pi / 2)
        # OpenCV saves hue values in uint8, so to fit 360 degrees in [0,255], every unit value is equivalent to 2
        # degrees and 180 is the maximum (2*180=360).

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame
        cv.imshow("Dense Optical Flow from Farneback", rgb)

        # Updates previous frame
        prev_gray = gray
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user
        # presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()

    return cost
