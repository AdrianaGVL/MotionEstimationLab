import cv2 as cv
import numpy as np
from scipy import signal
import time


def optical_flow(I1g, I2g, window_size, tau=1e-2):
    # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    w = (window_size - 1) // 2
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels

    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t,
                                                                                          boundary='symm', mode=mode)

    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here

            # if threshold τ is larger than the smallest eigenvalue of A'A=M (structure tensor):
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                # Least square fit to solve for unknown
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here; pinv: pseudo-inverse
                u[i, j] = nu[0]
                v[i, j] = nu[1]

    return u, v


def lkll_method(video):  # AGV modification
    cap = cv.VideoCapture('traffic.mp4')

    if not cap.isOpened():
        print("ERROR! Unable to open video or camera\n")

    ret, first_frame = cap.read()

    # Resize image and convert it to grayscale
    new_size = (320, 240)  # new_size=(width, height)
    first_frame = cv.resize(first_frame, new_size)
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

        # Resize image and convert it to grayscale
        new_size = (320, 240)
        frame = cv.resize(frame, new_size)
        # Opens a new window and displays the input frame
        cv.imshow("input", frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # AGV modification
        start = time.time_ns()

        # Calculates dense optical flow by Lucas-Kanade
        u, v = optical_flow(prev_gray, gray, window_size=15, tau=1e-2)

        # AGV modification
        end = time.time_ns()
        cost_ns = end - start
        cost = cost_ns / 1e9 if cost_ns >= 0 else 0

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(u, v)  # angle in radians

        # Dense optical flow representation: Sets image hue (channel 0) according to the optical flow direction
        # OpenCV saves hue values in uint8, so to fit 360 degrees in [0,255], every unit value is equivalent to 2
        # degrees and 180 is the maximum (2*180=360).
        mask[..., 0] = (angle * 180 / np.pi / 2)
        # Sets image value (channel 2) according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame
        cv.imshow("Dense Optical Flow from Lucas Kanade Low Level", rgb)

        # Updates previous frame
        prev_gray = gray
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user
        # presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    return cost  # AGV modification
