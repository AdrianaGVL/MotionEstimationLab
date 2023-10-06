import numpy as np
import cv2 as cv
import time
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import raft_small
from torchvision.utils import flow_to_image
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

plt.rcParams["savefig.bbox"] = "tight"


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


def raft_method(video):  # AGV modification
    frames, _, _ = read_video(str(video), output_format="TCHW")

    img1_batch = torch.stack([frames[50], frames[100]])
    img2_batch = torch.stack([frames[51], frames[101]])

    plot(img1_batch)
    plot(img2_batch)

    # weights = Raft_Large_Weights.DEFAULT
    weights = Raft_Small_Weights.DEFAULT

    transforms = weights.transforms()

    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
        img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
        return transforms(img1_batch, img2_batch)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # AGV modification
    start = time.time_ns()

    # model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)

    model = model.eval()

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

    # AGV modification
    end = time.time_ns()
    cost_ns = end - start
    cost = cost_ns / 1e9 if cost_ns >= 0 else 0

    print(f"type = {type(list_of_flows)}")
    print(f"length = {len(list_of_flows)} = number of iterations of the model")

    predicted_flows = list_of_flows[-1]
    print(f"dtype = {predicted_flows.dtype}")
    print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
    print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

    flow_imgs = flow_to_image(predicted_flows)

    # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
    img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

    grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
    plot(grid)

    for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
        # Note: it would be faster to predict batches of flows instead of individual flows
        img_copy = img1.clone()
        img1, img2 = preprocess(img1, img2)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        list_of_flows = model(img1.to(device), img2.to(device))
        predicted_flow = list_of_flows[-1][0]
        flow_img = flow_to_image(predicted_flow)

        # Opens a new window and displays the input frame
        img = cv.cvtColor(img_copy.to("cpu").permute(1, 2, 0).numpy(), cv.COLOR_RGB2BGR)
        cv.imshow("input", img)

        # Opens a new window and displays the output frame
        cv.imshow("Dense Optical Flow from RAFT model",
                  cv.cvtColor(flow_img.to("cpu").permute(1, 2, 0).numpy(), cv.COLOR_RGB2BGR))

        return cost  # AGV modification
