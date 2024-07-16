import numpy as np
from Segment_kmeans import *
import matplotlib.pyplot as plt
from PIL import Image
import os

# function to take all outputs per image and put it in 1 subplot
def combine_all_images(image_name_start):
    output_files = sorted([f for f in os.listdir("output") if f.startswith(image_name_start)])
    num_rows = int(np.ceil(len(output_files) / 3))
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    # loop through files in directory output and add them to subplot
    for i, filename in enumerate(output_files):
        im = imread(os.path.join("output", filename))
        axes[i].imshow(im)
        axes[i].set_title(filename[:-4])
        axes[i].axis('off')

    # output the figure
    plt.tight_layout()
    plt.savefig(os.path.join("output", image_name_start + "_all.jpg"))


# function to clear the output directory
def clear_output_directory():
    # loop through all files in directory output 
    for filename in os.listdir("output"):
        # delete file if it exists, else throw exception
        file_path = os.path.join("output", filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


# ----- Problem 1 -----
print("\n----- Problem 1 -----")

# 1c
image1_arr = np.array(Image.open("input/im1.jpg"))
image2_arr = np.array(Image.open("input/im2.jpg"))
image3_arr = np.array(Image.open("input/im3.png"))

K = [3,5,7]
Iters = [7,13,20]
R = [5,15,30]

clear_output_directory()

# image 1 run
for k in K:
    for iter in Iters:
        for r in R:
            image_out = Segment_kmeans(image1_arr, k, iter, r)
            output_filename = f"{"im1"}_K{k}_iters{iter}_R{r}.jpg"
            output_path = os.path.join("output", output_filename)
            imsave(output_path, image_out)
combine_all_images("im1")
print("Done with image 1")


# image 2 run
for k in K:
    for iter in Iters:
        for r in R:
            image_out = Segment_kmeans(image2_arr, k, iter, r)
            output_filename = f"{"im2"}_K{k}_iters{iter}_R{r}.jpg"
            output_path = os.path.join("output", output_filename)
            imsave(output_path, image_out)
combine_all_images("im2")
print("Done with image 2")


# image 3 run
for k in K:
    for iter in Iters:
        for r in R:
            image_out = Segment_kmeans(image3_arr, k, iter, r)
            output_filename = f"{"im3"}_K{k}_iters{iter}_R{r}.jpg"
            output_path = os.path.join("output", output_filename)
            imsave(output_path, image_out)
combine_all_images("im3")
print("Done with image 3")

