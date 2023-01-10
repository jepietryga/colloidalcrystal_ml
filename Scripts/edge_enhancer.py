# Purpose of this script is to rapidly enable comparisons between edges

import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import math
import copy
import pandas as pd
import os

import sys
sys.path.append("..")
from Utility.segmentation_utils import *
from PIL import Image


# Define possible images
im_dir = "/home/jacob/Desktop/Academics/Mirkin/colloidal_crystal_ML/Images/Additional/crystal size L3 L4 L5/L3"
im_list = glob.glob(os.path.join(im_dir,'*'))


interesting_files = ['../Images/Additional/L1 2.5, 5, 10 nM mixing assembly/L1 2.5 nM mixing /2.5nM_L1_inversion_01.tif',
                    "../Images/Additional/Si embed/2_01.tif"]
Alexa_files = glob.glob('../Images/Additional/Images for model (from Alexa)/*')
#print(Alexa_files)

# Note: 2_01.tif is really interesting
IS = ImageSegmenter(interesting_files[1],top_boundary=0,bottom_boundary=920,override_exists=False)

# gather images
threshold = IS.thresh
img2 = copy.deepcopy(IS.img2)

edge_filter_modes = ["None","Gaussian","Bilateral","Sharpen"]

# (gaussian_kernel,canny_tl,canny_tu,bilateral_d,bilateral_ss,use_bilateral)
canny_edge_modes = [
    ("Gaussian",[(3,3),80,160,80,80,False]), # Gaussian with kernel (3,3) and Canny 
    ("Bilateral", [(3,3),80,160,80,80,True]), # Bilateral (80,80) with Canny
]

subplot_shape = (2,2)
fig_dim = (32,18)
fig_edges, ax_edges = plt.subplots(*subplot_shape,figsize=fig_dim)
fig_hist, ax_hist = plt.subplots(*subplot_shape,figsize=fig_dim)
fig_dist, ax_dist = plt.subplots(*subplot_shape,figsize=fig_dim)
fig_bright_dark, ax_bright_dark = plt.subplots(*subplot_shape,figsize=fig_dim)
fig_watershed, ax_watershed = plt.subplots(*subplot_shape,figsize=fig_dim)


fig_edges.suptitle("Edges")
fig_hist.suptitle("Histograms")
fig_dist.suptitle("Distance Transform")
fig_bright_dark.suptitle("Bright-Dark Edges")
fig_watershed.suptitle("Watershed Markers")

# Walk through watershed process for each to evaluate
cem_mode = 0 # canny edge mode
for ii, efm in enumerate(edge_filter_modes):
    # Subplot access
    y = ii%2
    x = int(ii/2)

    
    # Make edges, then get edge intensities
    canny_edge = IS.canny_edge(*canny_edge_modes[cem_mode][1])
    if efm == "None":
        img_edges = img2*0
    elif efm == "Gaussian":
        img_edges = cv2.GaussianBlur(img2,(5,5),cv2.BORDER_DEFAULT)
    elif efm == "Bilateral":
        img_edges = cv2.bilateralFilter(img2,2*45,100,100)
    elif efm == "Sharpen":
        n = -1.
        m = 5
        kernel = np.array([[0,n,0],
                          [n,m,n],
                          [0,n,0]]
                          )*1

        img_edges = cv2.filter2D(img2,-1,kernel)
    
    img_edges[canny_edge == 0] = 0
    ax_oi = ax_edges[y,x]
    ax_oi.imshow(img_edges)
    ax_oi.title.set_text(efm)

    # Make histograms for brightness/darkness heuristic
    histogram, bin_edges = np.histogram(img_edges,bins=256)

    # remove 0 vals, 255 vals
    bin_edges = bin_edges[1:-1]
    histogram = histogram[1:-1]
    cut_off = bin_edges[np.argmax(histogram)]
    #cut_off = np.round(np.sum(bin_edges[:-1]*histogram/sum(histogram)))
    print(cut_off)
    ax_oi = ax_hist[y,x]
    ax_oi.title.set_text(efm)
    ax_oi.plot(bin_edges[:-1],histogram)
    ax_oi.plot([cut_off,cut_off],[0,np.max(histogram)],'--r')

    # Define bright edges
    bright_edges = copy.deepcopy(img_edges)
    bright_edges[bright_edges < cut_off] = 0
    bright_edges[bright_edges > 0] = 255

    # define dark edges
    dark_edges = copy.deepcopy(img_edges)
    dark_edges[dark_edges > cut_off] = 0
    dark_edges[dark_edges > 0] = 255

    # broaden edges for visibility in a plot, then plot it
    bright_broad = cv2.GaussianBlur(bright_edges,(3,3),cv2.BORDER_DEFAULT)
    dark_broad  = cv2.GaussianBlur(dark_edges,(3,3),cv2.BORDER_DEFAULT)
    color_img = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    color_img[bright_broad > 0] = (0,0,255)
    color_img[dark_broad > 0] = (0,255,0)
    #color_img[(bright_broad > 0) & (dark_broad>0)] = (255,0,0)

    ax_oi = ax_bright_dark[y,x]
    ax_oi.imshow(color_img)
    ax_oi.title.set_text(efm)

    # Perofrom watershed with new inforamtion
    bg_mark = cv2.dilate(threshold,(3,3),iterations=1)
    th = threshold-dark_edges
    dist = cv2.distanceTransform(th,cv2.DIST_L2,5)

    ax_oi = ax_dist[y,x]
    ax_oi.imshow(dist)
    ax_oi.title.set_text(efm)

    dist_mult = .25
    ret2, fg_mark = cv2.threshold(dist,dist_mult*dist.max(),255,0)
    fg_mark = np.uint8(fg_mark)

    unknown = cv2.subtract(bg_mark,fg_mark)

    outputs = cv2.connectedComponentsWithStats(fg_mark)
    label_increment = 1
    markers = outputs[1] + label_increment
    markers[unknown == 255] = 0
    markers2 = cv2.watershed(IS.img3,markers)

    ax_oi = ax_watershed[y,x]
    ax_oi.imshow(markers2)
    ax_oi.title.set_text(efm)
    ax_oi.text(-20,-20, f"Num region: {len(np.unique(markers2))}")



fig_edges.suptitle("Edges")
fig_hist.suptitle("Histograms")
fig_dist.suptitle("Distance Transform")
fig_bright_dark.suptitle("Bright-Dark Edges")
fig_watershed.suptitle("Watershed Markers")

fig_list = [("Edges",fig_edges),
            ("Histogram",fig_hist),
            ("Distance",fig_dist),
            ("Bright_Dark",fig_bright_dark),
            ("Watershed",fig_watershed)
]

for (name,fig) in fig_list:
    print(fig)
    fig.tight_layout()
    fig.savefig(f"{name}.png")
