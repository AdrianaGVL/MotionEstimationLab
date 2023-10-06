import cv2 as cv
import numpy as np
from scipy import signal

#Author: Adriana Galán Villamarín (AGV)

def cornersPoints(I1g, I2g, window_size, tau=1e-2):
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

    c = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here

            # if threshold τ is larger than the smallest eigenvalue of A'A=M (structure tensor):
            eigenvalues = np.min(abs(np.linalg.eigvals(np.matmul(A.T, A))))
            if eigenvalues >= tau:
                c[i, j] = eigenvalues
            else:
                c[i, j] = tau

    return c


def cornersMap(video, k=0.07, window_size=15, tau=1e-2, n=0):
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        print('Error opening the video')
        exit()

    ret, frame1 = cap.read()
    height, width, layers = frame1.shape
    gframe1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    cornerness_map_video = cv.VideoWriter('Cornerness_Response_Map.mp4', fourcc, 30.0, (width, height))

    nframe = 0

    while True:

        ret, frame2 = cap.read()
        gframe2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        if not ret:
            break

        c = cornersPoints(gframe1, gframe2, window_size, tau)
        cmap = cv.normalize(c, None, 0, 255, cv.NORM_MINMAX)
        cornerness_response_map = cv.applyColorMap(cmap.astype(np.uint8), cv.COLORMAP_JET)

        cornerness_map_video.write(cornerness_response_map)

        gframe1 = gframe2

    cap.release()
    cornerness_map_video.release()
    cv.destroyAllWindows()
