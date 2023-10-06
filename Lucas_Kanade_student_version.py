import cv2 as cv
import numpy as np
import time


def lucaskanadesv_method(video):  # AGV modification
    # Parameters for Shi-Tomasi corner detection. Similar to the popular Harris Corner Detector.
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Color to draw optical flow track
    color = (0, 255, 0)

    cap = cv.VideoCapture('traffic.mp4')
    if (not cap.isOpened()):
        print("ERROR! Unable to open video or camera\n")

    ret, first_frame = cap.read()

    # Resize image and convert it to grayscale
    new_size = (320, 240)
    first_frame = cv.resize(first_frame, new_size)
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Finds the strongest corners in the first frame by Shi-Tomasi method. We will track the optical flow for these corners. Similar to the popular Harris Corner Detector.
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Creates an image filled with zero intensities with the same dimensions as the frame for later drawing purposes
    mask = np.zeros_like(first_frame)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Frame empty!\n")
            break

        # Resize image and convert it to grayscale
        new_size = (320, 240)  # new_size=(width, height)
        frame = cv.resize(frame, new_size)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # AGV modification
        start = time.time_ns()

        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

        # Selects good feature points for previous position
        good_old = prev[status == 1]
        # Selects good feature points for next position
        good_new = next[status == 1]

        # AGV modification
        end = time.time_ns()
        cost_ns = end - start
        cost = cost_ns / 1e9 if cost_ns >= 0 else 0

        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness.
            mask = cv.line(mask, (np.int32(a), np.int32(b)), (np.int32(c), np.int32(d)), color,
                           2)  # np.int32() data conversion is required for the last version of OpenCV
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv.circle(frame, (np.int32(a), np.int32(b)), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv.add(frame, mask)

        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)

        # Opens a new window and displays the output frame
        cv.imshow("Sparse Motion Estimation from Lucas Kanade Student Version", output)

        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user
        # presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()

    return cost  # AGV modification
